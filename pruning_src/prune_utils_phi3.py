import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import os
import re
from collections import defaultdict
from .hf_utils import *
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer
from . import prune_method as lm_prune
'''
llama prune method
'''

@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None, H_w=None, use_weight=False, keep_num=0, return_H=False, scale=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.

    H_w是后一层权重的H
    """
    # Run GC and cleanup GPU memory
   

    def fusion_H(h_a, h_w, alpha=0.5):
        nor_a = torch.sqrt(torch.mean(h_a**2))
        nor_w = torch.sqrt(torch.mean(h_w**2))
        return h_a/nor_a*alpha + h_w/nor_w*(1-alpha)
    

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        if scale is not None:
            X_batch = X_batch.double().to(device='cuda')/(scale.view(1,1,-1)).to(device='cuda')
        else:
            X_batch = X_batch.double().to(device='cuda')
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch
    if use_weight:
        H = fusion_H(H, H_w, alpha=0)
    
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device='cuda')
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    r = torch.sum(eig_val[keep_num:])/torch.sum(eig_val)
    print(f"剪枝成分占比{r}")
    if return_H:
        return eig_val, eigen_vec, H
    else:
        del H
        return eig_val, eigen_vec
@torch.no_grad()
def get_best_Q(H, score, step=64, keep_num=0):
    d = torch.sqrt(torch.diag(H))
    H = H/d.view(1, -1)/d.view(-1, 1) 
    H = 1-H.abs()
    print(torch.sum(H<0.1))

    score = torch.pow(score, 1)
  
    num = 0
    keep_indices = None
    index_range = torch.tensor(range(H.shape[0])).cuda()
    while keep_num > num*step:
        if keep_indices is not None:
            mask_indices = torch.tensor(list(set(index_range.cpu().numpy()) - set(keep_indices.cpu().numpy()))).cuda()
            re_score = torch.min(H[keep_indices], dim=0) # 计算剩余部分和以保留部分的相关性得分
            score_ = score*re_score[0]
            score_ = score_[mask_indices]
            index_range_ = index_range[mask_indices]
        else:
            score_ = score
        t = step if (keep_num-num*step)>step else keep_num-num*step
        _, sorted_indices = torch.topk(score_.view(-1), k=t)
        if keep_indices is not None:
            sorted_indices = index_range_[sorted_indices]
        keep_indices = sorted_indices if keep_indices is None else torch.cat([keep_indices, sorted_indices], dim=0)
        num += 1
    mask_indices = torch.tensor([each for each in range(H.shape[0]) if each not in keep_indices.tolist()]).cuda()
    keep_indices = torch.cat([keep_indices, mask_indices], dim=0)
    return keep_indices
from tqdm import tqdm

class Pruner:
    def __init__(self,module,module_name=None, layers:LlamaDecoderLayer=None, logger=None):
        self.layer = layers
        self.module_name = module_name
        self.module = module
        self.dev = self.module.weight.device
        self.H = None
        self.nsamples = 0
        self.head_mask = 0
        self.logger = logger
        self.p_nsamples = 0  # preceding_nsamples
        self.p_H = None
        self.p_delta_H = None

        self.ffn_residual_map = None
        self.W = self.module.weight.data.float().t()
        self.delta_W = None
        ######### mlp和attn激活 #############
        self.attn_activation = {'q_out':[], 'k_out':[], 'v_out':[],'out_in':[]}
        self.mlp_activation = {'up_out':[], 'gate_out':[], 'down_in':[]}


    # calculate preceding weight's hessian and delta_hessian
    def preceding_add_batch(self, inp, ori_inps):
        # inp:[batch_size*seqlen,hidden_dim]

        inp = inp.float()
        count = self.p_nsamples
        ori_inp = ori_inps[count].to(self.dev)  # [seqlen,in]

        if len(ori_inp.shape)==3:
            ori_inp = ori_inp.squeeze(0)
        ori_inp = ori_inp.t().float()  # [in,seqlen]

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.module, nn.Linear):
            # inp: [batch_size,,seq_len,hidden_dim]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   #[batch_size*seq_len,hidden_dim]->[hidden_dim,batch_size*seq_len]
        else:
            raise  ValueError("we only support reconstruct linear layer")

        delta_inp = ori_inp - inp
        if self.p_H is None:
            self.columns = inp.shape[0]  # hidden_dim
            self.p_H = torch.zeros((self.columns, self.columns), device=self.dev).float()
            self.p_delta_H = torch.zeros((self.columns, self.columns), device=self.dev).float()

        self.p_H *= self.p_nsamples / (self.p_nsamples + tmp)
        self.p_delta_H *= self.p_nsamples / (self.p_nsamples + tmp)
        self.p_nsamples += tmp
        self.p_H += inp.matmul(inp.t()) * 2 / self.p_nsamples      # [hidden_dim,seq_len] [seq_len,hidden_dim]
        self.p_delta_H += inp.matmul(delta_inp.t()) * 2 / self.p_nsamples

    # calculate hessian matrix
    def add_batch(self, inp, ori_inps=None):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.module, nn.Linear):
            # inp: [batch_size,,seq_len,hidden_dim]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   #[batch_size*seq_len,hidden_dim]->[hidden_dim,batch_size*seq_len]
        else:
            raise  ValueError("we only support reconstruct linear layer")

        # inp:[batch_size*seqlen,hidden_dim]
        if self.H is None:
            self.columns = inp.shape[0]  # hidden_dim
            self.H = torch.zeros((self.columns, self.columns), device=self.dev).double()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.double()
        self.H += inp.matmul(inp.t())

    
    def get_mask_weight_adjust_lm(self,sparsity):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        percdamp=0.01
        # scale = get_scale(inps, soft_rate=0.1)
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        if predamp:
            H[diag, diag] += damp
        weight = self.module.weight.data.double()
        # dtype = self.module.weight.data.dtype

        
       
        count = self.layer.mlp.config.intermediate_size  # number of  neurons
        num_dropped = round(count * sparsity)
        
        # round to 8 for faster inference
       
        num_dropped = num_dropped - num_dropped%8

        if weight is not None:
            score = weight**2*torch.reshape(torch.diag(H), shape=(1, weight.shape[1]))
            score = torch.sum(score, dim=0)
        _, sorted_indices = torch.topk(score.view(-1), k=count)
        keep_num = count - num_dropped
        self.logger.info(f"---- MLP keepdim is {keep_num} ----")
        mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=H.dtype).cuda()
        mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
        
        # 使用基底表示剪枝部分
        Q_down = mask_Q.T.clone().cuda()
        H_ = H[sorted_indices[:keep_num]][:, sorted_indices[:keep_num]]  # 提取保留数据协方差矩阵
        H_inv = torch.inverse(H_)
        Ht =  H[sorted_indices[:keep_num]][:, sorted_indices[keep_num:]]
        Q_t = H_inv@Ht
        Q_down[:, sorted_indices[keep_num:]] = Q_t.clone()

        weight = weight@(mask_Q@Q_down).T

        self.module.weight.data = weight.half()

    
    def get_mask_weight_adjust_lm2(self,sparsity):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        
        weight = self.module.weight.data.double()
        # dtype = self.module.weight.data.dtype

        
       
        count = self.layer.mlp.config.intermediate_size  # number of  neurons
        num_dropped = round(count * sparsity)
        
        # round to 8 for faster inference
        num_dropped = num_dropped - num_dropped%8
        eig_val, Q, H = pca_calc(self.mlp_activation['down_in'],  keep_num=2048, return_H=True)
        
        # 选择合适的基底
        keep_num = count - num_dropped
        score = Q**2@torch.reshape(eig_val, shape=(Q.shape[1], 1))
        sorted_indices = get_best_Q(H, score.view(-1), keep_num=keep_num, step=16)
        
        self.logger.info(f"---- MLP2 keepdim is {keep_num} ----")
        mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=H.dtype).cuda()
        mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
        
        # 使用基底表示剪枝部分
        Q_down = mask_Q.T.clone().cuda()
        H_ = H[sorted_indices[:keep_num]][:, sorted_indices[:keep_num]]  # 提取保留数据协方差矩阵
        H_inv = torch.inverse(H_)
        Ht =  H[sorted_indices[:keep_num]][:, sorted_indices[keep_num:]]
        Q_t = H_inv@Ht
        Q_down[:, sorted_indices[keep_num:]] = Q_t.clone()

        weight = weight@(mask_Q@Q_down).T

        self.module.weight.data = weight.half()


    def get_mask_weight_adjust_pca(self,sparsity,num_consecutive=1,last_time_mask=None,percdamp=0.01,ffn_inps=None,ori_ffn_inps=None):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        # scale = get_scale(inps, soft_rate=0.1)
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        if predamp:
            H[diag, diag] += damp
        weight = self.module.weight.data.double()
        dtype = self.module.weight.data.dtype

        size = num_consecutive
        count = torch.sum(last_time_mask).item()  # number of heads or neurons
        num_dropped = round(count * sparsity)

        # round to 8 for faster inference
        if num_consecutive==1:
            num_dropped = num_dropped - num_dropped%8

        if weight is not None:
            score = weight**2*torch.reshape(torch.diag(H.double()), shape=(1, weight.shape[1]))
            score = torch.sum(score, dim=0)
        _, sorted_indices = torch.topk(score.view(-1), k=count)
        keep_num = count - num_dropped
       
        
        X_eig = torch.linalg.eigh(H)
    
        index = torch.argsort(X_eig[0], descending=True)
        eig_val = X_eig[0][index]
        eigen_vec = X_eig[1][:, index]
        mask_Q = eigen_vec[:, :keep_num]
        Q_down = mask_Q.T
        weight = weight@(mask_Q@Q_down).T

        self.module.weight.data[:,last_time_mask] = weight.to(dtype=dtype)

        
        tmp = last_time_mask.clone()
        tmp[last_time_mask] = torch.sum(mask_Q, dim=1)>0.5
        self.mask = tmp

    def get_mask_weight_adjust_iterpca_lm0(self,sparsity):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        # scale = get_scale(inps, soft_rate=0.1)
        percdamp=0.01
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        
        if predamp:
            H[diag, diag] += damp
        weight = self.module.weight.data.double()
        dtype = self.module.weight.data.dtype
        dev = weight.device
        count = self.layer.mlp.config.intermediate_size  # number of  neurons
        num_dropped = round(count * sparsity)

        # round to 8 for faster inference
        
        num_dropped = num_dropped - num_dropped%8

        if weight is not None:
            score = weight**2*torch.reshape(torch.diag(H.double()), shape=(1, weight.shape[1]))
            score = torch.sum(score, dim=0)
        _, sorted_indices = torch.topk(score.view(-1), k=count)
        keep_num = count - num_dropped
        self.logger.info(f"---- MLP keepdim is {keep_num} ----")
        mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=H.dtype).cuda()
        mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
        
        # 使用基底表示剪枝部分
        Q_down = mask_Q.T.clone().cuda()
        H_ = H[sorted_indices[:keep_num]][:, sorted_indices[:keep_num]]  # 提取保留数据协方差矩阵
        H_inv = torch.inverse(H_)
        Ht =  H[sorted_indices[:keep_num]][:, sorted_indices[keep_num:]]
        Q_t = H_inv@Ht
        Q_down[:, sorted_indices[keep_num:]] = Q_t.clone()

        

        
      
       
        ################ 迭代求解
      
        keep_rate = keep_num/self.layer.mlp.config.intermediate_size
        if keep_rate < 0.4: #如果剪枝率不够高，则不适用迭代求解
            self.logger.info(f'小于0.4，进行iterpca')
            # self.attn_activation = {'q_out':[], 'k_out':[], 'out_in':[]}
            # self.mlp_activation = {'up_out':[], 'gate_out':[], 'down_in':[]}
            Qa, Qb, Qc2 = lm_prune.LM_Iter_Join_PCA(mask_Q, Q_down, self.mlp_activation['down_in'], 
                                      self.mlp_activation['up_out'], 
                                      self.mlp_activation['gate_out'], keep_num)
        else:
            Qa, Qb, Qc2 = lm_prune.LM_Iter_Join_PCA(mask_Q, Q_down, self.mlp_activation['down_in'], 
                                      self.mlp_activation['up_out'], 
                                      self.mlp_activation['gate_out'], keep_num, simple=True)

        # 融合参数
        weight = weight@Qc2.T
        self.module.weight.data = weight.to(dtype=dtype)


        W_gate, W_up =  self.layer.mlp.gate_proj, self.layer.mlp.up_proj

        dtype = W_gate.weight.dtype
        W_ = W_gate.weight.to(device=dev, dtype=torch.float64)
        Qb = Qb.to(device=dev, dtype=torch.float64)
        W_gate.weight.data = torch.matmul(Qb.T, W_).to(device=dev, dtype=dtype)

        dtype = W_up.weight.dtype
        W_ = W_up.weight.to(device=dev, dtype=torch.float64)
        Qa = Qa.to(device=dev, dtype=torch.float64)
        W_up.weight.data = torch.matmul(Qa.T, W_).to(device=dev, dtype=dtype)

    def get_mask_weight_adjust_iterpca_lm(self,sparsity):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        # scale = get_scale(inps, soft_rate=0.1)
        percdamp=0.01
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        
        if predamp:
            H[diag, diag] += damp
        weight = self.module.weight.data.double()
        dtype = self.module.weight.data.dtype
        dev = weight.device
        count = self.layer.mlp.config.intermediate_size  # number of  neurons
        num_dropped = round(count * sparsity)

        # round to 8 for faster inference
        
        num_dropped = num_dropped - num_dropped%8

        if weight is not None:
            score = weight**2*torch.reshape(torch.diag(H.double()), shape=(1, weight.shape[1]))
            score = torch.sum(score, dim=0)
        _, sorted_indices = torch.topk(score.view(-1), k=count)
        keep_num = count - num_dropped
        self.logger.info(f"---- IterMLP2 keepdim is {keep_num} ----")
        mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=H.dtype).cuda()
        mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
        
        # 使用基底表示剪枝部分
        Q_down = mask_Q.T.clone().cuda()
        H_ = H[sorted_indices[:keep_num]][:, sorted_indices[:keep_num]]  # 提取保留数据协方差矩阵
        H_inv = torch.inverse(H_)
        Ht =  H[sorted_indices[:keep_num]][:, sorted_indices[keep_num:]]
        Q_t = H_inv@Ht
        Q_down[:, sorted_indices[keep_num:]] = Q_t.clone()

        ################ 迭代求解
      
        keep_rate = keep_num/self.layer.mlp.config.intermediate_size
        
        self.logger.info(f'进行iterpca')
            # self.attn_activation = {'q_out':[], 'k_out':[], 'out_in':[]}
            # self.mlp_activation = {'up_out':[], 'gate_out':[], 'down_in':[]}
        Qa, Qb, Qc2 = lm_prune.LM_Iter_Join_PCA2(sorted_indices, self.mlp_activation['down_in'], 
                                    self.mlp_activation['up_out'], 
                                    self.mlp_activation['gate_out'], keep_num)

        # 融合参数
        weight = weight@Qc2.T
        self.module.weight.data = weight.to(dtype=dtype)


        W_gate, W_up =  self.layer.mlp.gate_proj, self.layer.mlp.up_proj

        dtype = W_gate.weight.dtype
        W_ = W_gate.weight.to(device=dev, dtype=torch.float64)
        Qb = Qb.to(device=dev, dtype=torch.float64)
        W_gate.weight.data = torch.matmul(Qb.T, W_).to(device=dev, dtype=dtype)

        dtype = W_up.weight.dtype
        W_ = W_up.weight.to(device=dev, dtype=torch.float64)
        Qa = Qa.to(device=dev, dtype=torch.float64)
        W_up.weight.data = torch.matmul(Qa.T, W_).to(device=dev, dtype=dtype)


    def get_mask_weight_adjust_mha(self, sparsity):

        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        # scale = get_scale(inps, soft_rate=0.1)
        percdamp = 0.01
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        
        if predamp:
            H[diag, diag] += damp

        weight = self.module.weight.data.double()
        dtype = self.module.weight.data.dtype
        dev = weight.device
        count = self.layer.self_attn.hidden_size  # number of neurons
        num_dropped = round(count * sparsity)
        keep_ = count - num_dropped
        self.logger.info(f"---- MHA1 keepdim is {keep_} ----")
       
        eig_val, maskQ1, maskQ2, head_keep = lm_prune.MHA_PCA_LM(self.attn_activation['out_in'], keep_num=keep_, layer=self.layer)
        head_keep = torch.tensor(head_keep).cuda().to(torch.int)
        print(head_keep)
        ropeQ = lm_prune.ROPE_LM_PCA(self.attn_activation['q_out'], self.attn_activation['k_out'], None, None, False, head_keep, layer=self.layer)

        ### 融合 ###
        W = self.layer.self_attn.o_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=dev, dtype=torch.float64)
        W.weight.data = (W_@maskQ2.T@maskQ1.T).to(device=dev, dtype=torch.float16)
        if hasattr(self.layer.self_attn, 'rope_Q'):
            self.layer.self_attn.rope_Q = ropeQ.half().to(device=dev)
        else:
            assert ValueError('self_attn 模型没有"rope_Q"参数')

    def get_mask_weight_adjust_mha2(self, sparsity):
        '''
        舍去keep_head，保留的head使用相同的dim，便于加速，其次考虑weight，先剪枝rope，再调整attn out_project 的输入部分
        '''
        percdamp = 0.01
        last_time_mask = None
        # W:[out_dim,in_dim]
        # H:[in_dim,in_dim]
        # scale = get_scale(inps, soft_rate=0.1)
        H = self.H.double()
        predamp = True
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns)
        
        if predamp:
            H[diag, diag] += damp





        weight = self.module.weight.data.double()
        dtype = self.module.weight.data.dtype
        dev = weight.device
       
        count = self.layer.self_attn.hidden_size  # number of neurons
        num_dropped = round(count * sparsity)
        keep_ = count - num_dropped
        self.logger.info(f"---- MHA2 keepdim is {keep_} ----")
       

        
        
        ropeQ , maskQ1, maskQ2 = lm_prune.ROPE_MHA_PCA_LM(self.attn_activation['out_in'],self.attn_activation['q_out'], self.attn_activation['k_out'],self.attn_activation['v_out'],
                                                          weight=weight,
                                                          keep_num=keep_, layer=self.layer, logger=self.logger)
        ### 融合 ###
        W = self.layer.self_attn.o_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=dev, dtype=torch.float64)
        W.weight.data = (W_@maskQ2.T@maskQ1.T).to(device=dev, dtype=dtype)
        if hasattr(self.layer.self_attn, 'rope_Q'):
            self.layer.self_attn.rope_Q = ropeQ.half().to(device=dev)
        else:
            assert ValueError('self_attn 模型没有"rope_Q"参数')



    def free(self):
        self.H = None
        torch.cuda.empty_cache()



# def constrained_indicies(model,sorted_head_indicies,num_heads,\
#                              sorted_neuron_indicies,num_neurons,head_remain_ratio=0.1,neuron_remain_ratio=0.02):
#     device = sorted_head_indicies.data.device

#     num_hidden_layers, num_attention_heads, ffn_dim, hidden_size, attention_head_size = \
#         get_model_properties(model)

#     head_indicies = sorted_head_indicies[:num_heads]
#     neuron_indicies = sorted_neuron_indicies[:num_neurons]

#     head_idle_seat = {}
#     new_head_indicies = {}
#     neuron_idle_seat = {}
#     new_neuron_indicies = {}
#     # the max para can be  pruned
#     max_pruned_heads = math.ceil(num_attention_heads * (1-head_remain_ratio))
#     max_pruned_neurons = math.ceil(ffn_dim * (1-neuron_remain_ratio))

#     excess_heads = 0
#     excess_neurons = 0
#     for i in range(num_hidden_layers):
#         # head
#         head_low_bound = i * num_attention_heads - 1
#         head_up_bound = (i + 1) * num_attention_heads
#         bool_indicies = torch.gt(head_indicies,head_low_bound) & \
#                    torch.lt(head_indicies,head_up_bound)
#         pruned_heads = torch.sum(bool_indicies)
#         indicies = torch.nonzero(bool_indicies).squeeze(dim=1)
#         assert pruned_heads.item() == indicies.shape[0], "something wrong in constrained_indicies"
#         if pruned_heads > max_pruned_heads:
#             head_idle_seat[i] = 0
#             excess_num = pruned_heads - max_pruned_heads
#             excess_heads += excess_num
#             indicies = indicies[:max_pruned_heads]
#         else:
#             head_idle_seat[i] = max_pruned_heads - pruned_heads
#         new_head_indicies[i] = head_indicies[indicies]

#         # neuron
#         neuron_low_bound = i * ffn_dim - 1
#         neuron_up_bound = (i + 1) * ffn_dim
#         bool_indicies = torch.gt(neuron_indicies, neuron_low_bound) & \
#                    torch.lt(neuron_indicies,neuron_up_bound)
#         pruned_neurons = torch.sum(bool_indicies)
#         # # 保证是8的倍数有利于推理加速
#         # pruned_neurons = pruned_neurons - pruned_neurons%8

#         indicies = torch.nonzero(bool_indicies).squeeze(dim=1)
#         assert pruned_neurons.item() == indicies.shape[0], "something wrong in constrained_indicies"
#         if pruned_neurons  > max_pruned_neurons:
#             neuron_idle_seat[i] = 0
#             excess_num = pruned_neurons - max_pruned_neurons
#             excess_neurons += excess_num
#             indicies = indicies[:max_pruned_neurons]
#         else:
#             neuron_idle_seat[i] = max_pruned_neurons - pruned_neurons
#         new_neuron_indicies[i] = neuron_indicies[indicies]


#     new_head_candidate = defaultdict(list)
#     new_neuron_candidate = defaultdict(list)

#     while excess_heads>0:
#         prune_idx = sorted_head_indicies[num_heads]
#         idx = int(prune_idx/num_attention_heads)
#         if head_idle_seat[idx]>0:
#             new_head_candidate[idx].append(prune_idx)
#             head_idle_seat[idx] -= 1
#             excess_heads -= 1
#         num_heads += 1

#     while excess_neurons>0:
#         prune_idx = sorted_neuron_indicies[num_neurons]
#         idx = int(prune_idx/ffn_dim)
#         if neuron_idle_seat[idx]>0:
#             new_neuron_candidate[idx].append(prune_idx)
#             neuron_idle_seat[idx] -= 1
#             excess_neurons -= 1
#         num_neurons += 1

#     for i in range(num_hidden_layers):
#         if i in new_head_candidate:
#             new_head_indicies[i] = torch.cat((new_head_indicies[i],torch.tensor(new_head_candidate[i],device=device)),dim=0)
#         if i in new_neuron_candidate:
#             new_neuron_indicies[i] = torch.cat((new_neuron_indicies[i],torch.tensor(new_neuron_candidate[i],device=device)),dim=0)


#     head_indicies = torch.cat([new_head_indicies[key] for key in new_head_indicies],dim=0)
#     neuron_indicies = torch.cat([new_neuron_indicies[key] for key in new_neuron_indicies],dim=0)


#     return head_indicies,neuron_indicies


# def search_mask_by_param(model,importance=None,delete_para=None,layers_mask=None):

#     num_layers, num_attention_heads, ffn_dim, hidden_size, head_size = \
#         get_model_properties(model)

#     head_masks = layers_mask['head_mask']
#     neuron_masks = layers_mask['neuron_mask']
#     valid_head_masks = abs(head_masks)>1e-7
#     valid_neuron_masks = abs(neuron_masks)>1e-7

#     head_importance = importance['head']
#     neuron_importance = importance['neuron']

#     unpruned_head_importance = head_importance[valid_head_masks]
#     unpruned_neuron_importance = neuron_importance[valid_neuron_masks]

#     sorted_head_importance, sorted_head_indicies = unpruned_head_importance.view(-1).sort(descending=False)
#     sorted_neuron_importance, sorted_neuron_indicies = unpruned_neuron_importance.view(-1).sort(descending=False)

#     min_importance = float('inf')
#     head_remain_ratio = 0.2
#     neuron_remain_ratio = 0.2
#     max_prune_heads = int(torch.sum(valid_head_masks).item() * (1 - head_remain_ratio))
#     max_prune_neurons = int(torch.sum(valid_neuron_masks).item() * (1 - neuron_remain_ratio))

#     per_head_para = param_per_head(model,hidden_size, head_size)
#     per_neuron_para = param_per_neuron(model,hidden_size)

#     for num_heads in range(max_prune_heads + 1):
#         heads_param = per_head_para * num_heads
#         neurons_param = delete_para - heads_param
#         if neurons_param<0:
#             num_heads = int(delete_para/per_head_para)
#         num_neurons = int(neurons_param / per_neuron_para)
#         num_neurons = max(num_neurons, 0)
#         num_neurons = min(max_prune_neurons, num_neurons)

#         # 控制每层的剪枝率,避免整层崩溃
#         h_indicies, n_indicies = constrained_indicies(model, sorted_head_indicies, num_heads, \
#                                                       sorted_neuron_indicies, num_neurons,
#                                                       head_remain_ratio,neuron_remain_ratio)
#         total_importance = unpruned_head_importance[h_indicies].sum() + unpruned_neuron_importance[n_indicies].sum()
#         if total_importance < min_importance:
#             min_importance = total_importance
#             head_indicies = h_indicies
#             neuron_indicies = n_indicies

#         if neurons_param < 0:
#             break

#     new_head_mask = torch.ones(num_layers * num_attention_heads,dtype=torch.float16).cuda()
#     new_head_mask[~valid_head_masks.view(-1)] = 0.0
#     unpruned_head = new_head_mask[valid_head_masks.view(-1)]
#     unpruned_head[head_indicies] = 0.0
#     new_head_mask[valid_head_masks.view(-1)] = unpruned_head
#     new_head_mask = new_head_mask.view(num_layers, num_attention_heads)

#     new_neuron_mask = torch.ones(num_layers * ffn_dim,dtype=torch.float16).cuda()
#     new_neuron_mask[~valid_neuron_masks.view(-1)] = 0.0
#     unpruned_neuron = new_neuron_mask[valid_neuron_masks.view(-1)]
#     unpruned_neuron[neuron_indicies] = 0.0
#     new_neuron_mask[valid_neuron_masks.view(-1)] = unpruned_neuron
#     new_neuron_mask = new_neuron_mask.view(num_layers, ffn_dim)

#     return new_head_mask, new_neuron_mask


# def get_importance(args,model,dataloader,layers_mask,dev='cuda',ori_logits=False):
#     dataset = args.dataset
#     if isinstance(model, OPTForCausalLM):
#         model.model.decoder.gradient_checkpointing = True
#     elif isinstance(model, BloomForCausalLM):
#         model.transformer.gradient_checkpointing = True
#     elif isinstance(model, LlamaForCausalLM):
#         model.model.gradient_checkpointing = True
#     use_cache = model.config.use_cache
#     model.config.use_cache = False


#     for param in model.parameters():
#         param.requires_grad_(False)

#     full_head_mask = layers_mask['head_mask']
#     full_neuron_mask = layers_mask['neuron_mask']
#     full_head_mask.requires_grad_(True)
#     full_neuron_mask.requires_grad_(True)

#     neuron_handles = apply_neuron_mask(model, full_neuron_mask)

#     head_grads = []
#     neuron_grads = []
#     loss_fct = nn.CrossEntropyLoss()
#     ori_last_token_logits = []

#     nlls = []
#     nsamples = len(dataloader)
#     for inp, tgt in tqdm(dataloader,desc='getting mask importance'):
#         inp = inp.to(dev)
#         # tgt = tgt.to(dev)
#         lm_logits = model(inp, head_mask=full_head_mask).logits

#         # shift_logits: [batch_size,seq_len,num_word] [1,2047,50272]
#         shift_logits = lm_logits[:, :-1, :].contiguous()

#         # shift_label:[1,2047]
#         shift_labels = inp[:, 1:]

#         # shift_logits.view(-1, shift_logits.size(-1)): [2047,50272]  shift_labels.view(-1):[2047,]
#         shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#         shift_labels = shift_labels.view(-1).to(dev)
#         loss = loss_fct(shift_logits, shift_labels)

#         neg_log_likelihood = loss.double() * model.seqlen
#         nlls.append(neg_log_likelihood)

#         loss.backward()
#         head_grads.append(full_head_mask.grad.detach())
#         full_head_mask.grad = None
#         neuron_grads.append(full_neuron_mask.grad.detach())
#         full_neuron_mask.grad = None


#     # 检查ppl,保证输入正确
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
#     print(round(ppl.item(), 4))


#     for handle in neuron_handles:
#         handle.remove()

#     if isinstance(model, OPTForCausalLM):
#         model.model.decoder.gradient_checkpointing = False
#     elif isinstance(model, BloomForCausalLM):
#         model.transformer.gradient_checkpointing = False
#     elif isinstance(model, LlamaForCausalLM):
#         model.model.gradient_checkpointing = False
#     model.config.use_cache = use_cache

#     full_head_mask.requires_grad_(False)
#     full_neuron_mask.requires_grad_(False)
#     head_grads = torch.stack(head_grads, dim=0)
#     neuron_grads = torch.stack(neuron_grads, dim=0)

#     grads = {}
#     grads['head_grad'] = head_grads
#     grads['neuron_grad'] = neuron_grads
#     head_importance, neuron_importance = change_grad_to_saliency(grads)

#     importance = {}
#     importance['head'] = head_importance
#     importance['neuron'] = neuron_importance

#     has_inf = torch.isinf(head_importance).any()
#     has_nan = torch.isnan(head_importance).any()
#     assert not has_inf, "head_importance contains infinite values"
#     assert not has_nan, "head_importance contains NaN values"

#     has_inf = torch.isinf(neuron_importance).any()
#     has_nan = torch.isnan(neuron_importance).any()
#     assert not has_inf, "neuron_importance contains infinite values"
#     assert not has_nan, "neuron_importance contains NaN values"

#     if ori_logits:
#         ori_last_token_logits = torch.stack(ori_last_token_logits)

#     return importance,ori_last_token_logits


# def get_prune_ratios_and_masks(args,model,importance,layers_mask,para_to_change=None):


#     head_mask, neuron_mask = search_mask_by_param(model,importance=importance,\
#                                                   delete_para=para_to_change,layers_mask=layers_mask)

#     valid_head_mask = abs(layers_mask['head_mask']) > 0
#     valid_neuron_mask = abs(layers_mask['neuron_mask']) > 0
#     layers_head_remain_ratio = torch.sum(head_mask, dim=1) / torch.sum(valid_head_mask, dim=1)
#     layers_neuron_remain_ratio = torch.sum(neuron_mask, dim=1) / torch.sum(valid_neuron_mask, dim=1)

#     layers_prune_ratio = {}
#     layers_prune_ratio['head_prune_ratio'] = 1 - layers_head_remain_ratio
#     layers_prune_ratio['neuron_prune_ratio'] = 1 - layers_neuron_remain_ratio

#     new_layers_mask = {}
#     new_layers_mask['head_mask'] = head_mask.to(dtype=torch.float16)
#     new_layers_mask['neuron_mask'] = neuron_mask.to(dtype=torch.float16)

#     return layers_prune_ratio, new_layers_mask

# # for ablation
# def get_prune_mask_by_w_mag(args, model,layers_mask):

#     num_hidden_layers, num_heads, ffn_dim, hidden_size, head_size = \
#         get_model_properties(model)
#     prune_ratio = args.prune_ratio
#     attn_w_metric, ffn_w_metric = output_saliency_by_w_magnitude(model)
#     _, sorted_attn_indices = torch.sort(attn_w_metric, dim=1)  # 默认升序排列
#     _, sorted_ffn_indices = torch.sort(ffn_w_metric, dim=1)
#     ffn_zero = int(prune_ratio * ffn_dim)
#     head_zero = int(prune_ratio * num_heads)
#     no_int = False
#     if abs(prune_ratio * num_heads - head_zero) > 0.01:
#         no_int = True
#     for i in range(num_hidden_layers):
#         if no_int and i % 2 == 0:
#             ffn_indices_zero = sorted_ffn_indices[i, :ffn_zero + 1]
#             head_indices_zero = sorted_attn_indices[i, :head_zero + 1]
#         else:
#             ffn_indices_zero = sorted_ffn_indices[i, :ffn_zero]
#             head_indices_zero = sorted_attn_indices[i, :head_zero]

#         layers_mask['neuron_mask'][i, ffn_indices_zero] = 0
#         layers_mask['head_mask'][i, head_indices_zero] = 0

#     args.fixed_mask = 1

#     return layers_mask


# def fake_prune(model,layers_mask):
#     device = model.device
#     head_mask = layers_mask['head_mask']
#     neuron_mask = layers_mask['neuron_mask']
#     head_mask = head_mask.to(device)
#     neuron_mask = neuron_mask.to(device)

#     # head_mask = torch.zeros_like(head_mask)
#     # neuron_mask = torch.zeros_like(neuron_mask)

#     num_layers, num_heads, ffn_dim, hidden_size, head_size = \
#         get_model_properties(model)

#     if isinstance(model,LlamaForCausalLM):
#         num_key_value_heads = model.config.num_key_value_heads  # GQA
#         group_head_mask = torch.ones((num_layers,num_key_value_heads),dtype=torch.float16,device=device)
#         consecutive_heads = num_heads//num_key_value_heads
#         for i in range(num_layers):
#             for j in range(num_key_value_heads):
#                 q_group_head_mask = head_mask[i][j*consecutive_heads:(j+1)*consecutive_heads]
#                 if torch.all(abs(q_group_head_mask)<1e-5):
#                     group_head_mask[i][j] = 0

#     for i in range(num_layers):
#         Bq = None
#         Bk = None
#         Bv = None
#         if isinstance(model, OPTForCausalLM):
#             out_proj = model.model.decoder.layers[i].self_attn.out_proj
#             out_proj_w = out_proj.weight.data.view(-1, num_heads, head_size)
#             Wq = model.model.decoder.layers[i].self_attn.q_proj.weight.data.view(num_heads,head_size,-1)
#             Wk = model.model.decoder.layers[i].self_attn.k_proj.weight.data.view(num_heads,head_size,-1)
#             Wv = model.model.decoder.layers[i].self_attn.v_proj.weight.data.view(num_heads,head_size,-1)
#             if model.model.decoder.layers[i].self_attn.q_proj.bias is not None:
#                 Bq = model.model.decoder.layers[i].self_attn.q_proj.bias.data.view(num_heads, head_size)
#             if model.model.decoder.layers[i].self_attn.k_proj.bias is not None:
#                 Bk = model.model.decoder.layers[i].self_attn.k_proj.bias.data.view(num_heads, head_size)
#             if model.model.decoder.layers[i].self_attn.v_proj.bias is not None:
#                 Bv = model.model.decoder.layers[i].self_attn.v_proj.bias.data.view(num_heads, head_size)
#             fc1 = model.model.decoder.layers[i].fc1
#             fc2 = model.model.decoder.layers[i].fc2

#         elif isinstance(model, BloomForCausalLM):
#             out_proj = model.transformer.h[i].self_attention.dense
#             out_proj_w = out_proj.weight.data.view(-1, num_heads, head_size)
#             W = model.transformer.h[i].self_attention.query_key_value.weight.data.view(num_heads, 3, head_size,-1)
#             Wq = W[:, 0, ...]
#             Wk = W[:, 1, ...]
#             Wv = W[:, 2, ...]
#             if model.transformer.h[i].self_attention.query_key_value.bias is not None:
#                 bias = model.transformer.h[i].self_attention.query_key_value.bias.data.view(num_heads, 3, head_size)
#                 Bq = bias[:, 0, :]
#                 Bk = bias[:, 1, :]
#                 Bv = bias[:, 2, :]
#             fc1 = model.transformer.h[i].mlp.dense_h_to_4h
#             fc2 = model.transformer.h[i].mlp.dense_4h_to_h

#         elif isinstance(model,LlamaForCausalLM):
#             out_proj = model.model.layers[i].self_attn.o_proj
#             out_proj_w = out_proj.weight.data.view(-1,num_heads, head_size)
#             Wq = model.model.layers[i].self_attn.q_proj.weight.data.view(num_heads, head_size, -1)
#             Wk = model.model.layers[i].self_attn.k_proj.weight.data.view(num_key_value_heads, head_size, -1)
#             Wv = model.model.layers[i].self_attn.v_proj.weight.data.view(num_key_value_heads, head_size, -1)
#             if model.model.layers[i].self_attn.q_proj.bias is not None:
#                 Bq = model.model.layers[i].self_attn.q_proj.bias.data.view(num_heads, head_size)
#             if model.model.layers[i].self_attn.k_proj.bias is not None:
#                 Bk = model.model.layers[i].self_attn.k_proj.bias.data.view(num_key_value_heads, head_size)
#             if model.model.layers[i].self_attn.v_proj.bias is not None:
#                 Bv = model.model.layers[i].self_attn.v_proj.bias.data.view(num_key_value_heads, head_size)
#             fc11 = model.model.layers[i].mlp.up_proj
#             fc12 = model.model.layers[i].mlp.gate_proj
#             fc2 = model.model.layers[i].mlp.down_proj

#         '''
#             attention
#         '''
#         if not isinstance(model,LlamaForCausalLM):
#             # Wq:[num_heads,head_size,in_dim]  head_mask[i]:[num_heads,]
#             Wq = Wq * head_mask[i].view(num_heads, 1, 1)
#             Wk = Wk * head_mask[i].view(num_heads, 1, 1)
#             Wv = Wv * head_mask[i].view(num_heads, 1, 1)

#             # bias:[num_heads,head_size] head_mask[i]:[num_heads,]
#             if Bq is not None:
#                 Bq = Bq * head_mask[i].view(num_heads, 1)
#             if Bk is not None:
#                 Bk = Bk * head_mask[i].view(num_heads, 1)
#             if Bv is not None:
#                 Bv = Bv * head_mask[i].view(num_heads, 1)
#         else:
#             # Wq:[num_heads,head_size,in_dim]  head_mask[i]:[num_heads,]
#             Wq = Wq * head_mask[i].view(num_heads, 1, 1)
#             Wk = Wk * group_head_mask[i].view(num_key_value_heads, 1, 1)
#             Wv = Wv * group_head_mask[i].view(num_key_value_heads, 1, 1)

#             # bias:[num_heads,head_size] head_mask[i]:[num_heads,]
#             if Bq is not None:
#                 Bq = Bq * head_mask[i].view(num_heads, 1)
#             if Bk is not None:
#                 Bk = Bk * group_head_mask[i].view(num_key_value_heads, 1)
#             if Bv is not None:
#                 Bv = Bv * group_head_mask[i].view(num_key_value_heads, 1)

#         if isinstance(model, OPTForCausalLM):
#             model.model.decoder.layers[i].self_attn.q_proj.weight.data = Wq.view(-1,hidden_size)
#             model.model.decoder.layers[i].self_attn.k_proj.weight.data = Wk.view(-1,hidden_size)
#             model.model.decoder.layers[i].self_attn.v_proj.weight.data = Wv.view(-1,hidden_size)
#             if Bq is not None:
#                 model.model.decoder.layers[i].self_attn.q_proj.bias.data = Bq.view(-1)
#             if Bk is not None:
#                 model.model.decoder.layers[i].self_attn.k_proj.bias.data = Bk.view(-1)
#             if Bv is not None:
#                 model.model.decoder.layers[i].self_attn.v_proj.bias.data = Bv.view(-1)

#         elif isinstance(model, BloomForCausalLM):
#             W[:, 0, ...] = Wq
#             W[:, 1, ...] = Wk
#             W[:, 2, ...] = Wv
#             if Bq is not None:
#                 bias[:, 0, :] = Bq
#                 bias[:, 1, :] = Bk
#                 bias[:, 2, :] = Bv

#         elif isinstance(model,LlamaForCausalLM):
#             model.model.layers[i].self_attn.q_proj.weight.data = Wq.view(-1, hidden_size)
#             model.model.layers[i].self_attn.k_proj.weight.data = Wk.view(-1, hidden_size)
#             model.model.layers[i].self_attn.v_proj.weight.data = Wv.view(-1, hidden_size)
#             if Bq is not None:
#                 model.model.layers[i].self_attn.q_proj.bias.data = Bq.view(-1)
#             if Bk is not None:
#                 model.model.layers[i].self_attn.k_proj.bias.data = Bk.view(-1)
#             if Bv is not None:
#                 model.model.layers[i].self_attn.v_proj.bias.data = Bv.view(-1)

#         # out_proj_w:[out_dim,num_heads,head_size]
#         out_proj_w = out_proj_w * head_mask[i].view(1,num_heads,1)
#         out_proj.weight.data = out_proj_w.view(-1,num_heads*head_size)


#         '''
#            FFN
#         '''
#         if isinstance(model, OPTForCausalLM) or isinstance(model, BloomForCausalLM):
#             # fc1.weight.data:[out_dim,in_dim]  fc1.bias.data:[out_dim,]
#             fc1.weight.data = fc1.weight.data * neuron_mask[i].view(ffn_dim,1)
#             if fc1.bias is not None:
#                 fc1.bias.data = fc1.bias.data * neuron_mask[i].view(ffn_dim)

#             # fc2_w:[out_dim,in_dim] neuron_mask[i]:[ffn_dim]
#             fc2.weight.data = fc2.weight.data * neuron_mask[i].view(1,ffn_dim)

#         elif isinstance(model,LlamaForCausalLM):
#             # fc11.weight.data:[out_dim,in_dim]  fc11.bias.data:[out_dim,]
#             fc11.weight.data = fc11.weight.data * neuron_mask[i].view(ffn_dim, 1)
#             fc12.weight.data = fc12.weight.data * neuron_mask[i].view(ffn_dim, 1)
#             if fc11.bias is not None:
#                 fc11.bias.data = fc11.bias.data * neuron_mask[i].view(ffn_dim)
#             if fc12.bias is not None:
#                 fc12.bias.data = fc12.bias.data * neuron_mask[i].view(ffn_dim)
#             # fc2_w:[out_dim,in_dim] neuron_mask[i]:[ffn_dim]
#             fc2.weight.data = fc2.weight.data * neuron_mask[i].view(1, ffn_dim)

#     # model.model.decoder.layers[i].fc2.bias.data = torch.zeros_like(model.model.decoder.layers[i].fc2.bias.data)

#     return model


# def real_st_prune(model,layers_mask):
#     device = model.device

#     # head_mask = torch.zeros_like(head_mask)
#     # neuron_mask = torch.zeros_like(neuron_mask)

#     num_layers, num_heads, ffn_dim, hidden_size, head_size = \
#         get_model_properties(model)

#     # num_remain_heads = torch.sum(layers_mask['head_mask'],dtype=torch.float32)
#     # num_remain_neurons = torch.sum(layers_mask['neuron_mask'],dtype=torch.float32)
#     if layers_mask['head_mask'].dtype != torch.bool:
#         head_masks = abs(layers_mask['head_mask'])>1e-5
#         neuron_masks = abs(layers_mask['neuron_mask'])>1e-5
#         layers_mask['head_mask'] = head_masks
#         layers_mask['neuron_mask'] = neuron_masks
#     else:
#         head_masks = layers_mask['head_mask']
#         neuron_masks = layers_mask['neuron_mask']

#     # assert abs(num_remain_heads - torch.count_nonzero(head_masks))<1
#     # assert abs(num_remain_neurons - torch.count_nonzero(neuron_masks)) < 1,f'{abs(num_remain_neurons - torch.count_nonzero(neuron_masks))}'

#     if isinstance(model, OPTForCausalLM):
#         layers = model.model.decoder.layers
#     elif isinstance(model,LlamaForCausalLM):
#         layers = model.model.layers
#     elif isinstance(model,BloomForCausalLM):
#         layers = model.transformer.h

#     for i in range(num_layers):
#         layer = layers[i]
#         attn_mask = head_masks[i].cpu()
#         mlp_mask = neuron_masks[i].cpu()
#         if attn_mask is not None:
#             if isinstance(model, LlamaForCausalLM):
#                 # GQA
#                 num_key_value_heads = model.config.num_key_value_heads
#                 group_attn_mask = torch.ones(num_key_value_heads, device=device).bool()
#                 consecutive_heads = num_heads // num_key_value_heads
#                 if consecutive_heads != 1:
#                     for j in range(num_key_value_heads):
#                         q_group_head_mask = attn_mask[j * consecutive_heads:(j + 1) * consecutive_heads]
#                         if torch.all(q_group_head_mask == False):
#                             group_attn_mask[j] = False
#                     if torch.count_nonzero(group_attn_mask) == 0:
#                         tmp = random.randint(0, num_key_value_heads - 1)
#                         group_attn_mask[tmp] = True
#                         attn_mask[tmp * consecutive_heads:(tmp + 1) * consecutive_heads] = True

#                     group_repeat_times = []
#                     for j in range(num_key_value_heads):
#                         if group_attn_mask[j]:
#                             q_group_head_mask = attn_mask[j * consecutive_heads:(j + 1) * consecutive_heads]
#                             group_nonzero = torch.count_nonzero(q_group_head_mask)
#                             group_repeat_times.append(group_nonzero)
#                     group_repeat_times = torch.tensor(group_repeat_times, device=device)
#                 else:
#                     group_attn_mask = attn_mask

#                 retain_heads = torch.count_nonzero(attn_mask)
#                 remain_key_value_heads = torch.count_nonzero(group_attn_mask)
#                 attn_mask = attn_mask.repeat_interleave(head_size)
#                 group_attn_mask = group_attn_mask.repeat_interleave(head_size)
#                 # Prune the query, key and value projection weights
#                 # We reduce the size of the weights based on the attention mask
#                 layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
#                 layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(group_attn_mask)[0]]
#                 layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(group_attn_mask)[0]]
#                 if layer.self_attn.q_proj.bias is not None:
#                     layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[torch.where(attn_mask)[0]]
#                     layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data[torch.where(group_attn_mask)[0]]
#                     layer.self_attn.v_proj.bias.data = layer.self_attn.v_proj.bias.data[torch.where(group_attn_mask)[0]]

#                 # Update output dimensions of q, k, v projections based on remaining heads
#                 layer.self_attn.q_proj.out_features = attn_mask.sum().item()
#                 layer.self_attn.k_proj.out_features = group_attn_mask.sum().item()
#                 layer.self_attn.v_proj.out_features = group_attn_mask.sum().item()

#                 output_weight = layer.self_attn.o_proj.weight.data
#                 output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
#                 layer.self_attn.o_proj.weight.data = output_weight

#                 # Update layer configurations for the new output shape after pruning
#                 layer.self_attn.num_heads = retain_heads
#                 layer.self_attn.hidden_size = retain_heads * head_size  # wjt
#                 layer.self_attn.num_key_value_heads = remain_key_value_heads
#                 if consecutive_heads != 1:
#                     layer.self_attn.num_key_value_groups = group_repeat_times
#                 layer.self_attn.o_proj.in_features = attn_mask.sum().item()

#             elif isinstance(model, OPTForCausalLM):
#                 retain_heads = torch.count_nonzero(attn_mask)
#                 attn_mask = attn_mask.repeat_interleave(head_size)
#                 # Prune the query, key and value projection weights
#                 # We reduce the size of the weights based on the attention mask
#                 layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
#                 layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
#                 layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
#                 if layer.self_attn.q_proj.bias is not None:
#                     layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[torch.where(attn_mask)[0]]
#                     layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data[torch.where(attn_mask)[0]]
#                     layer.self_attn.v_proj.bias.data = layer.self_attn.v_proj.bias.data[torch.where(attn_mask)[0]]
#                 # Update output dimensions of q, k, v projections based on remaining heads
#                 layer.self_attn.q_proj.out_features = attn_mask.sum().item()
#                 layer.self_attn.k_proj.out_features = attn_mask.sum().item()
#                 layer.self_attn.v_proj.out_features = attn_mask.sum().item()

#                 output_weight = layer.self_attn.out_proj.weight.data
#                 output_weight = layer.self_attn.out_proj.weight.data[:, torch.where(attn_mask)[0]]
#                 layer.self_attn.out_proj.weight.data = output_weight

#                 # Update layer configurations for the new output shape after pruning
#                 layer.self_attn.num_heads = retain_heads
#                 layer.self_attn.embed_dim = retain_heads * head_size  # wjt
#                 layer.self_attn.out_proj.in_features = attn_mask.sum().item()


#             elif isinstance(model, BloomForCausalLM):
#                 retain_heads = torch.count_nonzero(attn_mask)
#                 # attn_mask = attn_mask.repeat_interleave(head_size)
#                 # Prune the query, key and value projection weights
#                 # We reduce the size of the weights based on the attention mask
#                 W = layer.self_attention.query_key_value.weight.data.view(num_heads, 3, head_size, -1)
#                 Wq = W[:, 0, ...]
#                 Wk = W[:, 1, ...]
#                 Wv = W[:, 2, ...]
#                 Wq = Wq[torch.where(attn_mask)[0]]
#                 Wk = Wk[torch.where(attn_mask)[0]]
#                 Wv = Wv[torch.where(attn_mask)[0]]
#                 W[torch.where(attn_mask)[0], 0, ...] = Wq
#                 W[torch.where(attn_mask)[0], 1, ...] = Wk
#                 W[torch.where(attn_mask)[0], 2, ...] = Wv
#                 layer.self_attention.query_key_value.weight.data = \
#                     W[torch.where(attn_mask)[0]].view(-1, hidden_size)
#                 if layer.self_attention.query_key_value.bias is not None:
#                     qkv_bias = layer.self_attention.query_key_value.bias.data.view(num_heads, 3, head_size)
#                     Bq = qkv_bias[:, 0, :]
#                     Bk = qkv_bias[:, 1, :]
#                     Bv = qkv_bias[:, 2, :]
#                     Bq = Bq[torch.where(attn_mask)[0]]
#                     Bk = Bk[torch.where(attn_mask)[0]]
#                     Bv = Bv[torch.where(attn_mask)[0]]
#                     qkv_bias[torch.where(attn_mask)[0], 0, :] = Bq
#                     qkv_bias[torch.where(attn_mask)[0], 1, :] = Bk
#                     qkv_bias[torch.where(attn_mask)[0], 2, :] = Bv
#                     layer.self_attention.query_key_value.bias.data = qkv_bias[torch.where(attn_mask)[0]].view(-1)

#                 # Update output dimensions of q, k, v projections based on remaining heads
#                 attn_mask = attn_mask.repeat_interleave(head_size)
#                 layer.self_attention.query_key_value.out_features = 3 * attn_mask.sum().item()

#                 output_weight = layer.self_attention.dense.weight.data
#                 output_weight = layer.self_attention.dense.weight.data[:, torch.where(attn_mask)[0]]
#                 layer.self_attention.dense.weight.data = output_weight

#                 # Update layer configurations for the new output shape after pruning
#                 layer.self_attention.num_heads = retain_heads
#                 layer.self_attention.hidden_size = retain_heads * head_size  # wjt
#                 layer.self_attention.dense.in_features = attn_mask.sum().item()

#         # MLP Weight Pruning
#         if mlp_mask is not None:
#             if isinstance(model, LlamaForCausalLM):
#                 # Prune the up and gate projection weights
#                 layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
#                 layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
#                 if layer.mlp.up_proj.bias is not None:
#                     layer.mlp.up_proj.bias.data = layer.mlp.up_proj.bias.data[torch.where(mlp_mask)[0]]
#                     layer.mlp.gate_proj.bias.data = layer.mlp.gate_proj.bias.data[torch.where(mlp_mask)[0]]

#                 # Update output dimensions of up and gate projections based on the mlp mask
#                 layer.mlp.up_proj.out_features = mlp_mask.sum().item()
#                 layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

#                 output_weight = layer.mlp.down_proj.weight.data
#                 layer.mlp.intermediate_size = mlp_mask.sum().item()
#                 output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]
#                 layer.mlp.down_proj.weight.data = output_weight

#                 layer.mlp.down_proj.in_features = mlp_mask.sum().item()

#             elif isinstance(model, OPTForCausalLM):
#                 # Prune the up and gate projection weights
#                 layer.fc1.weight.data = layer.fc1.weight.data[torch.where(mlp_mask)[0]]
#                 if layer.fc1.bias is not None:
#                     layer.fc1.bias.data = layer.fc1.bias.data[torch.where(mlp_mask)[0]]

#                 # Update output dimensions of up and gate projections based on the mlp mask
#                 layer.fc1.out_features = mlp_mask.sum().item()

#                 output_weight = layer.fc2.weight.data
#                 output_weight = layer.fc2.weight.data[:, torch.where(mlp_mask)[0]]
#                 layer.fc2.weight.data = output_weight

#                 layer.fc2.in_features = mlp_mask.sum().item()

#             elif isinstance(model, BloomForCausalLM):
#                 # Prune the up and gate projection weights
#                 layer.mlp.dense_h_to_4h.weight.data = layer.mlp.dense_h_to_4h.weight.data[torch.where(mlp_mask)[0]]
#                 if layer.mlp.dense_h_to_4h.bias is not None:
#                     layer.mlp.dense_h_to_4h.bias.data = layer.mlp.dense_h_to_4h.bias.data[torch.where(mlp_mask)[0]]

#                 # Update output dimensions of up and gate projections based on the mlp mask
#                 layer.mlp.dense_h_to_4h.out_features = mlp_mask.sum().item()

#                 output_weight = layer.mlp.dense_4h_to_h.weight.data
#                 output_weight = layer.mlp.dense_4h_to_h.weight.data[:, torch.where(mlp_mask)[0]]
#                 layer.mlp.dense_4h_to_h.weight.data = output_weight

#                 layer.mlp.dense_4h_to_h.in_features = mlp_mask.sum().item()


#     if isinstance(model, OPTForCausalLM):
#         for i in range(num_layers):
#             model.model.decoder.layers[i].st_pruned = True
#     elif isinstance(model, BloomForCausalLM):
#         for i in range(num_layers):
#             model.transformer.h[i].self_attention.st_pruned = True
#     elif isinstance(model, LlamaForCausalLM):
#         for i in range(num_layers):
#             model.model.layers[i].self_attn.st_pruned = True

#     model.config.st_pruned = True
