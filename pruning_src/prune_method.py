import logging
import os
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import time
import json
import torch.optim as optim 


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
    if return_H:
        return eig_val, eigen_vec, H
    else:
        del H
        return eig_val, eigen_vec


def LM_Iter_Join_PCA(Q1, Qt, inps, up_outs=None, gate_outs=None, keep_num=0, simple=False):
    '''
    LM_pca 和 iter_pca  结合
    '''
    if simple:
        Qa_, Qb_ = Q1.float(),Q1.float()
        Cs_recon = get_outs_list(Qa_, Qb_, gate_outs, up_outs)
        # Cs_recon2 = get_outs_list2(Abias, Bbias)
        Q_2 = get_pinverse_svd(Cs_recon, inps)
        # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
        Qc2 = Q_2.cpu()
        torch.cuda.empty_cache()  
    
        return Qa_.double().cuda() , Qb_.double().cuda() , Qc2.double().cuda() 
    else:
      # 剪枝率不够，直接忽略迭代pca
        n_step = int(1024**2*4/keep_num)
        if n_step > 1548:
            n_step = 2048
        else:
            n_step = 1024
        
        
        As_ = [a for a in up_outs]
        Bs_ = [a for a in gate_outs]
        Cs_ = [a for a in inps]


        # 原始lmpca方法
        Q1_init = Q1.float()
        Q2_init = Q1.float()
        Qt_init = Qt.T.float()
        # Qa2 = torch.zeros(size=(up_outs[0].shape[2], keep_num))
        # Qb2 = torch.zeros(size=(up_outs[0].shape[2], keep_num))
        Qa_, Qb_ = sequential_components2(As_, Bs_, Cs_, Q1_init, Q2_init, Qt_init, num_components=keep_num, n_step=n_step)
        # 加入head channel
        Cs_recon = get_outs_list(Qa_, Qb_, gate_outs, up_outs)
    
        # Cs_recon2 = get_outs_list2(Abias, Bbias)
        Q_2 = get_pinverse_svd(Cs_recon, inps)
        # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
        Qc2 = Q_2.cpu()
        torch.cuda.empty_cache()  
    
        return Qa_.double().cuda() , Qb_.double().cuda() , Qc2.double().cuda() 
    

@torch.no_grad()
def  LM_PCA(inps, keep_num):
    '''
    一种可以穿透非线性计算的近似PCA计算思路。
    一共分为两步，通过pca分析得到压缩矩阵Q，基于Q和协方差矩阵，选择基底。
    第二步，利用基底表示被剪枝部分。
    返回up_project和gate_project的剪枝maskQ，以及down的补偿融合矩阵Qdown
    '''
   
    H = None
    for X_batch in inps:
        X_batch = X_batch.double().cuda()
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H+H_batch
    # H = X_batch.mT @ X_batch
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    H[diag, diag] = H[diag, diag] + damp
    H = H.cuda()
    X_eig = torch.linalg.eigh(H)
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    Q = eigen_vec

    #
    
    # 选择合适的基底
    score = Q**2@torch.reshape(eig_val, shape=(Q.shape[1], 1))
    _, sorted_indices = torch.topk(score.view(-1), k=Q.shape[0])
    mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=torch.double).cuda()
    ss_indice, _ = torch.sort(sorted_indices[:keep_num])

    mask_Q[ss_indice, range(keep_num)] = 1  
    Q_down = mask_Q.T.clone().cuda()
    H_ = H[ss_indice][:, ss_indice]  # 提取保留数据协方差矩阵
    H_inv = torch.inverse(H_)
    Ht =  H[ss_indice][:, sorted_indices[keep_num:]]
    Q_t = H_inv@Ht


    # 计算down_project的融合矩阵
   
    Q_down[:, sorted_indices[keep_num:]] = Q_t.clone()
    

    return mask_Q, Q_down

def LM_Iter_Join_PCA2(sort_indice, Cs, As=None, Bs=None, keep_num=0, simple=False):
    '''
    LM_pca 和 iter_pca  结合
    '''
    ##my method
    head_keep =keep_num-1024
    head_indice = sort_indice[:head_keep]

    remain_indice = [k for k in range(As[0].shape[-1]) if k not in head_indice]
    print(f'remain shape = {len(remain_indice)}')
    As_ = [a[:,:,remain_indice] for a in As]
    Bs_ = [a[:,:,remain_indice] for a in Bs]
    Cs_ = [a[:,:,remain_indice] for a in Cs]
    Q12, Q_down2= LM_PCA(Cs_, keep_num=keep_num-head_keep)


    Q1_init = Q12.float()
    Q2_init = Q12.float()
    Qt_init = Q_down2.T.float()
 
    Qa = torch.zeros(size=(As[0].shape[2], keep_num))
    Qb = torch.zeros(size=(As[0].shape[2], keep_num))
    print(f'As_[0] shape = {As_[0].shape}')

    start_time = time.time()
    
    # Qt_init = Qs.float().clone()
    
    Qa_, Qb_ = sequential_components_iter(As_, Bs_, Cs_, Q1_init, Q2_init, Qt_init, num_components=keep_num-head_keep, n_step=1024)
    Qa[head_indice, range(head_keep)] = 1
    Qb[head_indice, range(head_keep)] = 1
    Qa[remain_indice, head_keep:] = Qa_
    Qb[remain_indice, head_keep:] = Qb_
    
    # Cs_recon = get_outs_list(Q12, Q12, Bs[:14], As[:14])
    Cs_recon = get_outs_list(Qa, Qb, Bs, As)
    # Cs_recon2 = get_outs_list2(Abias, Bbias)
    Q_2 = get_pinverse_svd(Cs_recon, Cs)
    # Qc = (Q[:, :num_components].double()@Q_2.T.cuda()).float().cpu()
    Qt = Q_2.cpu()
    print(f"diedai zhixing shijian {time.time()-start_time}")




    torch.cuda.empty_cache()  

    return Qa.double().cuda() , Qb.double().cuda() , Qt.double().cuda() 

def silu(x):
    return  torch.nn.functional.silu(x) 

def get_outs_list2(abias, bbias):
    output = [(a.cuda() * silu(b.cuda())).cpu() for a, b in zip(abias, bbias)]      
    return output

def get_outs_list_loss(Qa, Qb, Qc, gate_os, up_os, down_inps):
    output = [((up_o.float().cuda()@Qa.float().cuda() * silu(gate_o.float().cuda()@Qb.float().cuda()))@(Qc.float().cuda())).cpu() for gate_o, up_o in zip(gate_os, up_os)]  
    loss = [torch.nn.functional.mse_loss(out, target, reduction='mean')*1000 for out, target in zip(output, down_inps)]
    return torch.Tensor(loss).mean()

def get_out(Qa, Qb, Qt, gate_o, up_o, down_inps):
    for i in range(4):
        output = ((up_o[i].float().cuda()@Qa.float().cuda() * silu(gate_o.float().cuda()@Qb.float().cuda())).cpu())@Qt

    return output

def get_pinverse_svd(inps, target):
    '''
    求解X'@Q=X中的Q
    Q = inv(X'.T@X')@X'.T@X
    '''
    s = None
    s2 = None
    for index, batch in enumerate(inps):
        batch = batch.clone().cuda().double()
        x = target[index].clone().cuda().double()
        h = torch.sum(batch.mT@batch, dim=0)
        s = h if s is None else  s+h

        h = torch.sum(batch.mT@x, dim=0)
        s2 = h if s2 is None else  s2+h

        del batch
        del x
        del h


    
    damp = 0.01 * torch.mean(torch.diag(s))
    diag = torch.arange(s.shape[-1]).cuda()
    s[diag, diag] = s[diag, diag] + damp
    s_ = torch.inverse(s)
    return (s_@s2).double().cpu()

def compute_single_component(As, Bs, Cs, num_components, Abias, Bbias, max_epochs=50, lr=0.002):
    """求解单个主成分 qa, qb, qc"""
    n, _, d = Bs[0].shape
   
    # 初始化参数（小随机值）
    qb = torch.randn(d, num_components, device='cuda', requires_grad=True)
    qa = torch.randn(d, num_components, device='cuda', requires_grad=True)
    nn.init.normal_(qb, mean=0.0, std=0.00005)
    nn.init.normal_(qa, mean=0.0, std=0.00005)
  
    
    optimizer = optim.Adam([qa, qb], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    A, B, C= torch.cat(As, dim=0).cuda().float(), torch.cat(Bs, dim=0).cuda().float(), torch.cat(Cs, dim=0).cuda().float()
    a_bias = torch.cat(Abias, dim=0).cuda().float()
    b_bias = torch.cat(Bbias, dim=0).cuda().float()
    for epoch in range(max_epochs):
        random_index = random.sample(range(A.shape[0]), 32)
        # 轮流冻结参数
        qa.requires_grad = True
        qb.requires_grad = True
        optimizer.zero_grad()
    
        # 计算当前近似
        # for i in range(2):
        #     C_approx = ((A[random_index[i*16:i*16+16]] @ qa+a_bias[random_index[i*16:i*16+16]]) * (silu(B[random_index[i*16:i*16+16]] @ qb+b_bias[random_index[i*16:i*16+16]]))) 
        #     loss = torch.nn.functional.mse_loss(C[random_index[i*16:i*16+16]], C_approx, reduction='mean')*num_components
        #     # 反向传播
        #     loss.backward()

       
        C_approx = ((A[random_index] @ qa+a_bias[random_index]) * (silu(B[random_index] @ qb+b_bias[random_index]))) 
        loss = torch.nn.functional.mse_loss(C[random_index], C_approx, reduction='mean')*num_components
            # 反向传播
        loss.backward()
        optimizer.step()
      
        # 打印训练信息
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        if (epoch+1) %10 ==0:
             scheduler.step()
        # 早停检查
    return qa.detach().cpu(), qb.detach().cpu()



def weighted_mse_loss(input, target):
    # 计算绝对误差作为权重
    mean = torch.mean(torch.abs(target))
    weights = torch.abs(target)>3*mean +1  # 使用target的绝对值作为权重
    # 也可以使用其他权重策略，比如: weights = torch.abs(target) + 1e-6 (避免零权重)
    
    # 计算平方误差
    squared_errors = (input - target) ** 2
    
    # 应用权重
    weighted_errors = weights * squared_errors
    
    # 返回均值
    return torch.mean(weighted_errors)

def compute_single_component2(As, Bs, Cs, Q1_init, Q2_init, Qt,  num_components, Abias, Bbias, max_epochs=100, lr=0.001):
    """求解单个主成分 qa, qb, qc"""
    n, _, d = Bs[0].shape
    device = Bs[0].device
    # print(Q1_init.is_leaf)
    # print(Q2_init.is_leaf)
    # 初始化参数（小随机值）
    # if not Q1_init.is_leaf:
    #     Q1_init = Q1_init.detach().clone().requires_grad_(True).cuda()
    # if not Q2_init.is_leaf:
    #     Q2_init = Q2_init.detach().clone().requires_grad_(True).cuda()
    # print(Q1_init.is_leaf)
    # print(Q2_init.is_leaf)
    qa = torch.empty_like(Q1_init).copy_(Q1_init).requires_grad_(True).cuda()
    qa = torch.nn.Parameter(qa)
    qb = torch.empty_like(Q2_init).copy_(Q2_init).requires_grad_(True).cuda()
    qb = torch.nn.Parameter(qb)
  
    # print(qa.is_leaf)
    # print(qb.is_leaf)
    
  
    # nn.init.normal_(qb, mean=0.0, std=0.0005)
    # nn.init.normal_(qa, mean=0.0, std=0.0005)
    # nn.init.normal_(qc, mean=0.0, std=0.05)
    # qa.data = torch.nn.functional.normalize(qa.data, p=2, dim=0)
    # qb.data = torch.nn.functional.normalize(qb.data, p=2, dim=0)
    
    optimizer = optim.Adam([qa, qb], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)

    prev_loss = float('inf')

    A, B, C= torch.cat(As, dim=0).cuda().float(), torch.cat(Bs, dim=0).cuda().float(), torch.cat(Cs, dim=0).cuda().float()
    print(len(A))
    a_bias = torch.cat(Abias, dim=0).cuda().float()
    b_bias = torch.cat(Bbias, dim=0).cuda().float()
    for epoch in range(max_epochs):
        random_index = random.sample(range(A.shape[0]), 32)

        # A_ = A[random_index].cuda()
        # B_ = B[random_index].cuda()
        # C_ = C[random_index].cuda()
        # a_ = a_bias[random_index].cuda()
        # b_ = b_bias[random_index].cuda()
        # 轮流冻结参数
        qa.requires_grad = True
        qb.requires_grad = True
        optimizer.zero_grad()
    
        # 计算当前近似
        # C_approx = (( A_@ qa+a_) * (silu(B_ @ qb+b_))) @qc
        # loss = torch.nn.functional.mse_loss(C_, C_approx, reduction='mean')*num_components
        # loss = weighted_mse_loss(C_approx, C_)*num_components

        
        C_approx = ((A[random_index] @ qa+a_bias[random_index]) * (silu(B[random_index] @ qb+b_bias[random_index]))) 
        loss = torch.nn.functional.mse_loss(C[random_index], C_approx, reduction='mean')*num_components
        # loss = weighted_mse_loss(C_approx, C[random_index])*num_components
        # 反向传播
        loss.backward()
        optimizer.step()
      
        # 打印训练信息
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        if (epoch+1) % 20 ==0:
             scheduler.step()
        # 早停检查
    return qa.detach().cpu(), qb.detach().cpu()

def get_outs_list(Qa, Qb, gate_os, up_os):
    # output = [((up_o.double().cuda()@Qa.double().cuda() * silu(gate_o.double().cuda()@Qb.double().cuda()))@(Qt.T.double().cuda())).cpu() for gate_o, up_o in zip(gate_os, up_os)]  
    output = [(((up_o.double().cuda()@Qa.double().cuda()).half() * silu((gate_o.double().cuda()@Qb.double().cuda())).half())).cpu() for gate_o, up_o in zip(gate_os, up_os)]      
    return output

def sequential_components(As, Bs, Cs, H, sort_indice, num_components=3, n_step= 1024, rest_rate=0.2):
    """逐个计算主成分并剥离残差"""
   
    D = As[0].shape[2]
    if num_components > int(D*rest_rate):
        head_keep = int((num_components - int(D*rest_rate))/(1-rest_rate))
        pca_keep  = num_components - head_keep
    else:
        head_keep = 0
        pca_keep  = num_components - head_keep
   
    Qa = torch.zeros(size=(As[0].shape[2], num_components))
    Qb = torch.zeros(size=(As[0].shape[2], num_components))
    
    head_indice = sort_indice[:head_keep]
    print(f'head_keep is {head_keep}')
  

    
    Abias = [torch.zeros(size=(x.shape[0], x.shape[1], pca_keep)) for x in As]
    Bbias = [torch.zeros(size=(x.shape[0], x.shape[1], pca_keep)) for x in Bs]

    # 获取剩余部分的投影矩阵
    X_eig = torch.linalg.eigh(H[sort_indice[head_keep:]][:, sort_indice[head_keep:]])
    index = torch.argsort(X_eig[0], descending=True)
    eigen_vec = X_eig[1][:, index]
    Q = eigen_vec
  
    C_residuals = [(c[:, :, sort_indice[head_keep:]].cuda().double()@Q[:, :pca_keep].cuda().double()).cpu().double() for c in Cs]
    inti_lr = 0.001
    del Q, eigen_vec, X_eig, index
    torch.cuda.empty_cache()
    ss1 = time.time()
    for k in range((D - head_keep)//n_step):
        print(f"\nComputing component {k+1}...")
        As_ = [x[:,:, sort_indice[k*n_step+head_keep:k*n_step+n_step+head_keep]  ] for x in As]
        Bs_ = [x[:,:, sort_indice[k*n_step+head_keep:k*n_step+n_step+head_keep]] for x in Bs]
        # Cs_ = [x[:,:,indice] for x in Cs]
        ss2 = time.time()
        with torch.enable_grad():
            qa, qb = compute_single_component(As_, Bs_, C_residuals, pca_keep, Abias, Bbias, lr=inti_lr*0.5**k)
        print(f'{k}_step更新的时间:{time.time()-ss2}')
        Abias = [(x.cuda().float()@qa.cuda() + a.cuda()).cpu() for x, a in zip(As_, Abias)]
        Bbias = [(x.cuda().float()@qb.cuda() + a.cuda()).cpu() for x, a in zip(Bs_, Bbias)]
        Qa[sort_indice[k*n_step+head_keep:k*n_step+n_step+head_keep], head_keep:] = qa
        Qb[sort_indice[k*n_step+head_keep:k*n_step+n_step+head_keep], head_keep:] = qb
    k = (D - head_keep)//n_step+1
    if (D - head_keep)%n_step > 0: #  有多余的
        print(f"\nComputing component for rest {(D - head_keep) - (D - head_keep)//n_step*n_step}...")
        step = (D - head_keep) - (D - head_keep)//n_step*n_step

        As_ = [x[:,:,sort_indice[-step:]] for x in As]
        Bs_ = [x[:,:,sort_indice[-step:]] for x in Bs]
        # Cs_ = [x[:,:,indice] for x in Cs]
        with torch.enable_grad():
            qa, qb = compute_single_component(As_, Bs_, C_residuals, pca_keep, Abias, Bbias, lr=inti_lr*0.5**k)
        Abias = [(x.cuda().float()@qa.cuda() + a.cuda()).cpu() for x, a in zip(As_, Abias)]
        Bbias = [(x.cuda().float()@qb.cuda() + a.cuda()).cpu() for x, a in zip(Bs_, Bbias)]
        Qa[sort_indice[-step:], head_keep:] = qa
        Qb[sort_indice[-step:], head_keep:] = qb
    # 填补head_indice
    Qa[head_indice, range(head_keep)] = 1
    Qb[head_indice, range(head_keep)] = 1

    print(f'迭代更新的总时间：{time.time()-ss1}')
    ss = time.time()
    Cs_recon = get_outs_list(Qa, Qb, Bs, As)
    # Cs_recon2 = get_outs_list2(Abias, Bbias)
    Q_2 = get_pinverse_svd(Cs_recon, Cs)
    # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
    Qc = Q_2.cpu()
    print(f'补偿计算时间：{time.time()-ss}')
      
    
    return Qa, Qb, Qc


def get_pinverse_svd2(inps):
    '''
    求解X'@Q=X中的Q
    Q = inv(X'.T@X')@X'.T@X
    '''
    s = inps@inps.T
  
    damp = 0.01 * torch.mean(torch.diag(s))
    diag = torch.arange(s.shape[-1]).cuda()
    s[diag, diag] = s[diag, diag] + damp
    s_ = torch.inverse(s)
   
    return (inps.T@s_).float()

def sequential_components2(As, Bs, Cs, Q1, Q2, Qt, num_components=3, n_step= 1024):
    """逐个计算主成分并剥离残差"""
   
   
    Qa = torch.zeros(size=(As[0].shape[2], num_components))
    Qb = torch.zeros(size=(As[0].shape[2], num_components))
    
    step = num_components

    
    Abias = [torch.zeros(size=(x.shape[0], x.shape[1], num_components)) for x in As]
    Bbias = [torch.zeros(size=(x.shape[0], x.shape[1], num_components)) for x in Bs]

    Qt_inverse = get_pinverse_svd2(Qt[:, :num_components].cuda().T)

    # I = Qt[:, :num_components].cuda().T@Qt_inverse

  
    C_residuals = [(c.cuda().float()@Qt_inverse.cuda().float()).cpu().float() for c in Cs]
  

    # del Qt_inverse

    torch.cuda.empty_cache()
  
    # C_residuals = [(c.cuda().float()@Qt.cuda().float()).cpu().float() for c in Cs]
    inti_lr = 0.001

    ss1 = time.time()
    for k in range(As[0].shape[2]//n_step):
        print(f"\nComputing component {k+1}...")
        As_ = [x[:,:,k*n_step:k*n_step+n_step] for x in As]
        Bs_ = [x[:,:,k*n_step:k*n_step+n_step] for x in Bs]
        # Cs_ = [x[:,:,indice] for x in Cs]
        Q1_init = Q1[k*n_step:k*n_step+n_step]
        Q2_init = Q2[k*n_step:k*n_step+n_step]
        ss2 = time.time()
        with torch.enable_grad():
            qa, qb = compute_single_component2(As_, Bs_, C_residuals, Q1_init, Q2_init, Qt, num_components, Abias, Bbias, lr=inti_lr*0.5**k)
        print(f'{k}_step更新的时间:{time.time()-ss2}')
        Abias = [(x.cuda().float()@qa.cuda() + a.cuda()).cpu() for x, a in zip(As_, Abias)]
        Bbias = [(x.cuda().float()@qb.cuda() + a.cuda()).cpu() for x, a in zip(Bs_, Bbias)]

        torch.cuda.empty_cache()
        # Cs_recon2 = get_outs_list2(Abias, Bbias)
        # Q_2 = get_pinverse_svd(Cs_recon2, C_residuals)
        # # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
        # C_residuals = [0.1*(c.cuda().float()@Q_2.T.cuda().float()).cpu().float()+0.9*c for c in C_residuals]


        Qa[k*n_step:k*n_step+n_step] = qa
        Qb[k*n_step:k*n_step+n_step] = qb
    k = As[0].shape[2]//n_step
    if As[0].shape[2]%n_step > 0: #  有多余的
        print(f"\nComputing component for rest {As[0].shape[2] - As[0].shape[2]//n_step*n_step}...")
        step = As[0].shape[2] - As[0].shape[2]//n_step*n_step
        As_ = [x[:,:,-step:] for x in As]
        Bs_ = [x[:,:,-step:] for x in Bs]
        Q1_init = Q1[-step:]
        Q2_init = Q2[-step:]
        # Cs_ = [x[:,:,indice] for x in Cs]
        with torch.enable_grad():
            qa, qb = compute_single_component2(As_, Bs_, C_residuals, Q1_init,Q2_init, Qt, num_components, Abias, Bbias, lr=inti_lr*0.5**k)
        Abias = [(x.cuda().float()@qa.cuda() + a.cuda()).cpu() for x, a in zip(As_, Abias)]
        Bbias = [(x.cuda().float()@qb.cuda() + a.cuda()).cpu() for x, a in zip(Bs_, Bbias)]
        Qa[-step:] = qa
        Qb[-step:] = qb
    print(f'迭代更新的总时间：{time.time()-ss1}')
    # ss = time.time()
    # # Cs_recon = get_outs_list(Qa, Qb, Bs, As)
    # Cs_recon2 = get_outs_list2(Abias, Bbias)
    # Q_2 = get_pinverse_svd(Cs_recon2, Cs)
    # # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
    # Qc = Q_2.float().cpu()
    # print(f'补偿计算时间：{time.time()-ss}')
    # torch.cuda.empty_cache()  
    
    return Qa, Qb#, Qc



def sequential_components_iter(As, Bs, Cs, Q1, Q2, Qt, num_components=3, n_step= 1024):
    """逐个计算主成分并剥离残差"""
   
   
    Qa = torch.zeros(size=(As[0].shape[2], num_components))
    Qb = torch.zeros(size=(As[0].shape[2], num_components))
    
    step = num_components

    
    Abias = [torch.zeros(size=(x.shape[0], x.shape[1], num_components)) for x in As]
    Bbias = [torch.zeros(size=(x.shape[0], x.shape[1], num_components)) for x in Bs]

    Qt_inverse = get_pinverse_svd2(Qt[:, :num_components].cuda().T)

    # I = Qt[:, :num_components].cuda().T@Qt_inverse

  
    C_residuals = [(c.cuda().double()@Qt_inverse.cuda().double()).cpu().float() for c in Cs]
    # C_residuals = [(c.cuda().double()@Qt.cuda().double()).cpu().float() for c in Cs]

    # del Qt_inverse

    torch.cuda.empty_cache()
  
    # C_residuals = [(c.cuda().double()@Qt.cuda().double()).cpu().float() for c in Cs]
    inti_lr = 0.001

    com_step = int(n_step*num_components/As[0].shape[2])

    ss1 = time.time()
    leiji_step = 0
    for k in range(As[0].shape[2]//n_step):
        print(f"\nComputing component {k+1}...")
        As_ = [x[:,:,k*n_step:k*n_step+n_step] for x in As]
        Bs_ = [x[:,:,k*n_step:k*n_step+n_step] for x in Bs]
        # Cs_ = [x[:,:,indice] for x in Cs]
        com_step = torch.sum(Q1[:k*n_step+n_step]>0.5)
        Q1_init = Q1[k*n_step:k*n_step+n_step, leiji_step:com_step]
        Q2_init = Q2[k*n_step:k*n_step+n_step, leiji_step:com_step]
        ss2 = time.time()
        with torch.enable_grad():
            qa, qb = compute_single_component2(As_, Bs_, [c[:,:,leiji_step:com_step] for c in C_residuals], Q1_init, Q2_init, None, num_components, 
                                          [c[:,:,leiji_step:com_step] for c in Abias], [c[:,:,leiji_step:com_step] for c in Bbias], lr=inti_lr*0.5**k)
        print(f'{k}_step更新的时间:{time.time()-ss2}')
        # for jj in range(len(As_)):
        #     Abias[jj][:, :, :com_step] = Abias[jj][:, :, :com_step] + (As_[jj].cuda().float()@qa.cuda()).cpu()
        #     Bbias[jj][:, :, :com_step] = Bbias[jj][:, :, :com_step] + (Bs_[jj].cuda().float()@qb.cuda()).cpu()
       

        torch.cuda.empty_cache()
        # Cs_recon2 = get_outs_list2(Abias, Bbias)
        # Q_2 = get_pinverse_svd(Cs_recon2, C_residuals)
        # # Qc = (Q[:, :num_components].double()@Q_2.T.cuda()).float().cpu()
        # C_residuals = [0.1*(c.cuda().double()@Q_2.T.cuda().double()).cpu().float()+0.9*c for c in C_residuals]


        Qa[k*n_step:k*n_step+n_step,leiji_step:com_step] = qa
        Qb[k*n_step:k*n_step+n_step,leiji_step:com_step] = qb
        leiji_step = com_step
    k = As[0].shape[2]//n_step
    if As[0].shape[2]%n_step > 0: #  有多余的
        print(f"\nComputing component for rest {As[0].shape[2] - As[0].shape[2]//n_step*n_step}...")
        step = As[0].shape[2] - As[0].shape[2]//n_step*n_step
        As_ = [x[:,:,-step:] for x in As]
        Bs_ = [x[:,:,-step:] for x in Bs]
        Q1_init = Q1[-step:, leiji_step:num_components]
        Q2_init = Q2[-step:, leiji_step:num_components]
        com_step = num_components
        # Cs_ = [x[:,:,indice] for x in Cs]
        # qa, qb = compute_single_component(As_, Bs_, C_residuals,Q1_init,Q2_init, num_components, Abias, Bbias, lr=inti_lr*0.5**k)
        with torch.enable_grad():
            qa, qb = compute_single_component2(As_, Bs_, [c[:,:,leiji_step:com_step] for c in C_residuals], Q1_init, Q2_init, None, num_components, 
                                          [c[:,:,leiji_step:com_step] for c in Abias], [c[:,:,leiji_step:com_step] for c in Bbias], lr=inti_lr*0.5**k)
        # Abias = [(x.cuda().float()@qa.cuda() + a.cuda()).cpu() for x, a in zip(As_, Abias)]
        # Bbias = [(x.cuda().float()@qb.cuda() + a.cuda()).cpu() for x, a in zip(Bs_, Bbias)]
        Qa[-step:,leiji_step:num_components] = qa
        Qb[-step:,leiji_step:num_components] = qb
    print(f'迭代更新的总时间：{time.time()-ss1}')
    # ss = time.time()
    # # Cs_recon = get_outs_list(Qa, Qb, Bs, As)
    # Cs_recon2 = get_outs_list2(Abias, Bbias)
    # Q_2 = get_pinverse_svd(Cs_recon2, Cs)
    # # Qc = (Q[:, :num_components].float()@Q_2.T.cuda()).float().cpu()
    # Qc = Q_2.float().cpu()
    # print(f'补偿计算时间：{time.time()-ss}')
    # torch.cuda.empty_cache()  
    
    return Qa, Qb#, Qc


######################################## attention prune #############################################

################################## MHA_PCA_LM ###################################

def get_score(keep_head, del_head, score, keep_num):
    sum_score = 0
    for i in range(score.shape[0]):
        if i in del_head:
            continue
        elif i in keep_head:
            sum_score += torch.sum(score[i])
        else:
            sum_score += torch.sum(score[i, :keep_num])
    return sum_score


@torch.no_grad()
def MHA_PCA_LM(
    X: list[torch.Tensor],  ignore_masks: list[torch.Tensor] | None = None, keep_num=0, layer=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用与mha的pca, 判断是否保留整个头
    步骤：step1 pca分解，获得Q，计算每个channel的幅值得分
         step2 按照幅值得分，排列，并筛选出keep_num个chanel
         step3 统计每个head的channel的保留情况，保留数超过127的，默认全部保留
         step4 计算每个head的pca，并统计每个head每个投影的幅值得分
         step5 按照每个head每个投影的幅值得分，计算最佳全保留、全删除、均匀剪枝的分配结果
         step6 补偿

    """
    head_nums = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.float().cuda()
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch
   
    # 
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    Hc = H.clone()
    Hc[diag, diag] = Hc[diag, diag] + damp
    X_eig = torch.linalg.eigh(Hc.cuda().double())
    del Hc
    score = X_eig[1]**2@torch.reshape(X_eig[0], shape=(H.shape[1], 1))
    _, sorted_indices = torch.topk(score.view(-1), k=H.shape[0])
    mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=torch.double)
    mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
    
    h = torch.sum(mask_Q, dim=1)
    head_num = torch.sum(h.view(head_nums, -1), dim=1).to(int)  # 每个头保留的channel数量
    

    # 计算单个头的pca
    Q = torch.zeros_like(H).cuda()
    val = torch.zeros_like(H[0]).cuda()
    for i in range(head_nums):
        X_eig = torch.linalg.eigh(H[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim])
        index = torch.argsort(X_eig[0], descending=True)
        eig_val = X_eig[0][index]
        eigen_vec = X_eig[1][:, index]
        Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = eigen_vec
        val[i*head_dim:i*head_dim+head_dim] = eig_val/torch.sum(eig_val)  # 归一化特征值，用于后续头保留

    # 计算哪些头全部保留，哪些头全部删除，哪些头均匀压缩
    score = torch.reshape(val, shape=(head_nums, head_dim))*torch.sum(score.view(head_nums, head_dim), dim=1, keepdim=True)
    keep_num_h = int(keep_num/head_nums)
    all_score = torch.sum(score)
    print(f'all score is {all_score}')
    head_score = torch.sum(score, dim=1)
    # 均匀压缩
    mean_score = torch.sum(score[:, :keep_num_h])
    print(f'mean compression score is {mean_score}')
    #
    opt_score = 0
    for i in range(head_nums):
        opt_score += torch.sum(score[i, :head_num[i]])
    print(f'optimal score is {opt_score}')
    # 自适应压缩，全保留，均匀压缩，全删除
    print(torch.sort(head_num))
    keep_head = [i  for i in range(head_nums) if head_num[i] >= (head_dim-1)]
    del_head = []
    
    remain_head = [i for i in range(head_nums) if i not in keep_head and i not in del_head]
    if head_nums - len(keep_head)-len(del_head) == 0:  # 全部保留
        keep_num_ = 0

    else:
        keep_num_ = int((keep_num - len(keep_head)*head_dim)/(head_nums - len(keep_head)-len(del_head)))

        print(f'step 1:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')
        score1 = get_score(keep_head, del_head, score, keep_num_)
        print(f'adaptive score1 is {score1}')
        sum_score = score1

        for i in range(head_nums):
            if i not in remain_head:
                continue
            # 先计算是否可以全部剪枝
            index = remain_head[int(torch.argmin(head_score[remain_head]).cpu().item())]
            if head_nums - 1 - len(keep_head)-len(del_head) == 0: #这种情况出现在，保留的维度不能很好的均分，然后除了全部保留，和全部删除，只有一个头被用来调整，此时不能调
                continue
            k_now = int((keep_num - len(keep_head)*head_dim)/(head_nums-1 - len(keep_head)-len(del_head)))
            if k_now > head_dim :  # 这种情况出现在，如果remain中有两个，且keep_num大于一般的head_dim，此时删除一个，就会导致出现k_now大于head_dim的情况
                continue
            cur_score = get_score(keep_head, del_head+[index], score, k_now)
            if cur_score > sum_score:
                # print(f'cur_score is {cur_score}')
                del_head.append(index)
                sum_score = cur_score
                keep_num_ = k_now
                remain_head =  [e for e in remain_head if e != index]
                print(f'step update del head append:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')

            
            # 计算是否可以全部保留
            if head_nums - 1 - len(keep_head)-len(del_head) == 0: #这种情况出现在，保留的维度不能很好的均分，然后除了全部保留，和全部删除，只有一个头被用来调整，此时不能调
                continue
            index = remain_head[int(torch.argmax(head_score[remain_head]).cpu().item())]
            k_now = int((keep_num - len(keep_head)*head_dim-head_dim)/(head_nums-1 - len(keep_head)-len(del_head)))
            if k_now <=0 :  # 保留的channel数量，不足一个head_dim，这种情况出现在remain head 保留维度比较小的时候，所有remain head的保留维度加一起不足一个head_dim
                continue
        
            cur_score = get_score(keep_head+[index], del_head, score, k_now)
            if cur_score > sum_score:
                # print(f'cur_score is {cur_score}')
                keep_head.append(index)
                sum_score = cur_score
                keep_num_ = k_now
                remain_head =  [e for e in remain_head if e != index]
                print(f'step update keep head append:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')




    print(f'keep haeds are {keep_head}, del head are {del_head},  keep_num form {keep_num_h} to {keep_num_}')
    keep_num_h = keep_num_
    new_Q  = torch.zeros(H.shape).cuda().to(dtype=H.dtype)
    for i in range(head_nums):
        if i in del_head:
            continue
        if i in keep_head:
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim]
        else:
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim]
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim+keep_num_h:i*head_dim+head_dim] = 0

    Q = new_Q
    M = Q@Q.T
    H_ = M.T@H@M
    damp = 0.01 * torch.mean(torch.diag(H_))
    diag = torch.arange(H_.shape[-1]).cuda()
    H_[diag, diag] = H_[diag, diag] + damp
    H_inv = torch.inverse(H_)
    
    # del H_
    # Q_t = None  # 计算keep_data到mask_data的表示矩阵
    # for X_batch in X:
    #     X_batch = X_batch.cuda()
    #     X_batch = X_batch.float()
      
    #     q = torch.sum(H_inv@(X_batch@M).mT@X_batch, dim=0)
    #     Q_t = q if Q_t is None else Q_t+q

    Q_t = H_inv@M.T@H


    # 计算每个头的keep_num 用于rope剪枝
    keep_head_num = []
    for i in range(head_nums):
        if i in del_head:
            keep_head_num.append(0)
        elif i in keep_head:
            keep_head_num.append(head_dim)
        else:
            keep_head_num.append(keep_num_h)


    return X_eig[0], new_Q, new_Q.T@Q_t, keep_head_num

################################## ROPE_PCA_LM ###################################
@torch.no_grad()
def ROPE_LM_PCA(qs, ks,  ignore_masks: list[torch.Tensor] | None = None, H_w=None, use_weight=False, keep_num=0,layer=None, sparsity=2, group=1):
    '''
    一种可以穿透rope的，2值稀疏pca方法
    两个步骤：利用qk点乘幅值，对每个channel打分；其次依次进行稀疏化，转换成可以融合的二值稀疏压缩矩阵
    '''
    Qs = []
    head_nums = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    if group==1:
        for i in range(head_nums):
            q_out_x = [x[:, i].cuda() for x in qs]
            k_out_x = [x[:, i].cuda() for x in ks]
            qk_out_x = [q*k for q,k in zip(q_out_x, k_out_x)]
            # eig_val, Q, H = pca_calc(q_out_x+k_out_x, None, H_w, use_weight, keep_num)

            # 计算qk幅值
            qk_mean = None
            for X_batch in qk_out_x:
                X_batch = X_batch.double().cuda()
                m = X_batch.abs().mean(dim=(0, 1)).cuda().detach()
                qk_mean = m if qk_mean is None else qk_mean+m 

            # Q = sparse_pca_LM_qk(q_out_x, k_out_x, qk_mean, keep_num=keep_num[i], sparsity=sparsity)

            k = int(keep_num[i]/2)
            qk_mean = qk_mean[:head_dim//2]+qk_mean[head_dim//2:]
            _, sorted_indices = torch.topk(qk_mean.view(-1), k=64)
            keep_indices = sorted_indices[:k]
            Q = torch.zeros(size=(128 ,128)).cuda()
            for j in range(head_dim//2):
                if j in keep_indices:
                    Q[j, j] = 1
                    Q[j+head_dim//2, j+head_dim//2] = 1
            Qs.append(Q.view(1, head_dim ,head_dim))
        Qs = torch.cat(Qs, dim=0).unsqueeze(0).cuda()
    else:
        qk_mean = None
        group_k = max(keep_num)  # 应该是一个group中最大的那个
        for i in range(head_nums):
            q_out_x = [x[:, i].cuda() for x in qs]
            k_out_x = [x[:, i].cuda() for x in ks]
            qk_out_x = [q*k for q,k in zip(q_out_x, k_out_x)]
            # eig_val, Q, H = pca_calc(q_out_x+k_out_x, None, H_w, use_weight, keep_num)
            # 计算qk幅值
            if keep_num[i]>0:
                for X_batch in qk_out_x:
                    X_batch = X_batch.double().cuda()
                    m = X_batch.abs().mean(dim=(0, 1)).cuda().detach()
                    qk_mean = m if qk_mean is None else qk_mean+m 
            # Q = sparse_pca_LM_qk(q_out_x, k_out_x, qk_mean, keep_num=keep_num[i], sparsity=sparsity)
            if (i+1)%group == 0 :  # gropus
                Q = torch.zeros(size=(head_dim ,head_dim)).cuda()
                if qk_mean is not None:
                    k = int(group_k/2)
                    qk_mean = qk_mean[:head_dim//2]+qk_mean[head_dim//2:]
                    _, sorted_indices = torch.topk(qk_mean.view(-1), k=64)
                    keep_indices = sorted_indices[:k]
                    
                    for j in range(head_dim//2):
                        if j in keep_indices:
                            Q[j, j] = 1
                            Q[j+head_dim//2, j+head_dim//2] = 1
                    Qs.append(Q.view(1, head_dim ,head_dim).expand(group, -1,-1))
                    qk_mean=None
                else:
                    Qs.append(Q.view(1, head_dim ,head_dim).expand(group, -1,-1))
        Qs = torch.cat(Qs, dim=0).unsqueeze(0).cuda()
    return Qs

def sparse_pca_LM_qk(q, k, qk_mean, keep_num, sparsity, weight=None,kechuantou=False):
   
    RH = None
    for X_batch in q+k:
        X_batch = X_batch.double().cuda()
        
        if weight is not None:
            weight = weight.double().cuda()
            H_batch = torch.sum(weight@X_batch.mT @ X_batch@weight, dim=0)
        else:
            H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        RH = H_batch if RH is None else RH+H_batch
   
    # H = X_batch.mT @ X_batch
    damp = 0.01 * torch.mean(torch.diag(RH))
    diag = torch.arange(RH.shape[-1]).cuda()
    RH[diag, diag] = RH[diag, diag] + damp
    RH = RH.cuda()
    _, sorted_indices = torch.sort(qk_mean.view(-1))
    channel_score = qk_mean.cuda().clone().view(-1, 1)
    V =torch.zeros(size=(RH.shape[1], RH.shape[0])).to(dtype=RH.dtype, device=RH.device)

    for j in range(int(RH.shape[1]//sparsity)):
        if sparsity==2: #   施加可穿透
            # support = torch.arange(j*sparsity, j*sparsity+sparsity) # 依次选择channel
            support = torch.tensor([j, j+64], dtype=int) # llama系列rope选取
        else:
            support = sorted_indices[j*sparsity: j*sparsity+sparsity]

        # support = torch.range(0, 4096)
        ri = support.view(-1, 1).repeat(1, sparsity)
        ci = support.view(1, -1).repeat(sparsity, 1)
        # Step 4: 求解子集的最大特征向量
        sub_H = RH[ri, ci]  # 提取支持集对应的子矩阵
        X_eig = torch.linalg.eigh(sub_H)
        eig_val = X_eig[0]
        eigen_vec = X_eig[1]
        channel_score[support] = torch.sum(channel_score[support])*torch.sqrt(eig_val.view(-1, 1)/torch.sum(eig_val))

        if sparsity == 2:
            if eigen_vec[0,0]*eigen_vec[1, 1] - eigen_vec[0,1]*eigen_vec[1, 0] < 0:
                eigen_vec[:, 1] *= -1
            # print('xiuzheng')
        # print(f'index为：{support.cpu().numpy()}, 先前得分为：{score[support].cpu().numpy()},稀疏后得分：{channel_score[support].cpu().numpy()}')
        V[ri, ci]  = eigen_vec

     # ###################### 重新计算qk绝对值的均值
    q = [q_.double().cuda()@V for q_ in q]
    k = [k_.double().cuda()@V for k_ in k]
    qk_out_x = [(q*k).abs() for q,k in zip(q, k)]
    qk_mean = None
    for X_batch in qk_out_x:
        X_batch = X_batch.double().cuda()
        m = X_batch.abs().mean(dim=(0, 1)).cuda().detach()
        qk_mean = m if qk_mean is None else qk_mean + m
    channel_score = qk_mean.cuda().clone().view(-1, 1)
    ################################

    _, sorted_indices = torch.topk(channel_score.view(-1), k=RH.shape[1])
    V1 = V[:, sorted_indices]

    V1[:, keep_num:] = 0
    

    return V1




###################################### ROPE_MHA_LM ###############################
def get_recon_out(q,k,v,del_heads,rope_Q=None):
    '''
    qkv shape is (b, 32, 2048, 128)
    '''
    
    if rope_Q is not None:
        q = q.double().cuda()@rope_Q.double().cuda()
        k = k.double().cuda()@rope_Q.double().cuda()
    bsz, head_num, q_len ,head_dim= q.size()
    v[:, del_heads] = 0

    attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.cuda().half(),
            k.cuda().half(),
            v.cuda().half(),
            attn_mask=None,
            dropout_p=0,
            is_causal=True,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, head_dim*head_num)

    return attn_output.cpu()


@torch.no_grad()
def ROPE_MHA_PCA_LM(X: list[torch.Tensor], q, k, v, weight=None, ignore_masks: list[torch.Tensor] | None = None, keep_num=0, layer=None, logger=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    qkv的shape=(b, )
    应用与mha的pca, 判断是否保留整个头
    步骤：step1 pca分解，获得Q，计算每个channel的幅值得分
         step2 按照幅值得分，排列，并筛选出keep_num个chanel
         step3 统计每个head的channel的保留情况，保留数超过127的，默认全部保留
         step4 计算每个head的pca，并统计每个head每个投影的幅值得分
         step5 按照每个head每个投影的幅值得分，计算最佳全保留、全删除、均匀剪枝的分配结果
         step6 补偿
    """

    ########## 
    head_nums = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    groups = layer.self_attn.num_key_value_groups

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double().cuda()
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch
        del X_batch
   
    # 
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    
    Hc = H.clone()
    Hc[diag, diag] = Hc[diag, diag] + damp
    X_eig = torch.linalg.eigh(Hc.cuda().double())
    del Hc
    # score = X_eig[1]**2@torch.reshape(X_eig[0], shape=(H.shape[1], 1))
    if weight is not None:
        score = weight.cuda()**2*torch.reshape(torch.diag(H.double()), shape=(1, weight.shape[1]))
        score = torch.sum(score, dim=0)

    _, sorted_indices = torch.topk(score.view(-1), k=H.shape[0])
    mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=torch.double)
    mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
    
    h = torch.sum(mask_Q, dim=1)
    head_num = torch.sum(h.view(head_nums, -1), dim=1).to(int)  # 每个头保留的channel数量
    

    # 计算单个头的pca
    Q = torch.zeros_like(H).cuda()
    val = torch.zeros_like(H[0]).cuda()
    for i in range(head_nums):
        X_eig = torch.linalg.eigh(H[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim])
        index = torch.argsort(X_eig[0], descending=True)
        eig_val = X_eig[0][index]
        eigen_vec = X_eig[1][:, index]
        Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = eigen_vec
        val[i*head_dim:i*head_dim+head_dim] = eig_val/torch.sum(eig_val)  # 归一化特征值，用于后续头保留
    
    # 计算哪些头全部保留，哪些头全部删除，哪些头均匀压缩
    score = torch.reshape(val, shape=(head_nums, head_dim))*torch.sum(score.view(head_nums, head_dim), dim=1, keepdim=True)
    keep_num_h = int(keep_num/head_nums)
    all_score = torch.sum(score)
    print(f'all score is {all_score}')
    head_score = torch.sum(score, dim=1)
    # 均匀压缩
    mean_score = torch.sum(score[:, :keep_num_h])
    print(f'mean compression score is {mean_score}')
    #
    opt_score = 0
    for i in range(head_nums):
        opt_score += torch.sum(score[i, :head_num[i]])
    print(f'optimal score is {opt_score}')
    # 自适应压缩，全保留，均匀压缩，全删除
    print(torch.sort(head_num))
    # keep_head = [i  for i in range(head_nums) if head_num[i] >= (head_dim-16)]
    keep_head = []
    del_head = [i  for i in range(head_nums) if head_num[i] < int(keep_num_h//4)] # 小于平均维度的1/4，则认为需要删除
    # del_head = []
    remain_head = [i for i in range(head_nums) if i not in keep_head and i not in del_head]
    if head_nums - len(keep_head)-len(del_head) == 0:  # 全部保留
        keep_num_ = 0

    else:
        keep_num_ = int((keep_num - len(keep_head)*head_dim)/(head_nums - len(keep_head)-len(del_head)))

        # print(f'step 1:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')
        score1 = get_score(keep_head, del_head, score, keep_num_)
        # print(f'adaptive score1 is {score1}')
        sum_score = score1

        for i in range(head_nums):
            if i not in remain_head:
                continue
            # 先计算是否可以全部剪枝
            index = remain_head[int(torch.argmin(head_score[remain_head]).cpu().item())]
            if head_nums - 1 - len(keep_head)-len(del_head) == 0: #这种情况出现在，保留的维度不能很好的均分，然后除了全部保留，和全部删除，只有一个头被用来调整，此时不能调
                continue
            k_now = int((keep_num - len(keep_head)*head_dim)/(head_nums-1 - len(keep_head)-len(del_head)))
            if k_now > head_dim :  # 这种情况出现在，如果remain中有两个，且keep_num大于一般的head_dim，此时删除一个，就会导致出现k_now大于head_dim的情况
                continue
            cur_score = get_score(keep_head, del_head+[index], score, k_now)
            if cur_score > sum_score:
                # print(f'cur_score is {cur_score}')
                del_head.append(index)
                sum_score = cur_score
                keep_num_ = k_now
                remain_head =  [e for e in remain_head if e != index]
                # print(f'step update del head append:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')

        ## 限制keep_num_必须大于原始维度的3/4， 也就是96
        if keep_num_ < 96:
            for i in range(head_nums):
                if i not in remain_head:
                    continue
                # 先计算是否可以全部剪枝
                index = remain_head[int(torch.argmin(head_score[remain_head]).cpu().item())]
                if head_nums - 1 - len(keep_head)-len(del_head) == 0: #这种情况出现在，保留的维度不能很好的均分，然后除了全部保留，和全部删除，只有一个头被用来调整，此时不能调
                    continue
                k_now = int((keep_num - len(keep_head)*head_dim)/(head_nums-1 - len(keep_head)-len(del_head)))
                if k_now > head_dim :  # 这种情况出现在，如果remain中有两个，且keep_num大于一般的head_dim，此时删除一个，就会导致出现k_now大于head_dim的情况
                    continue
                
                del_head.append(index)
                keep_num_ = k_now
                remain_head =  [e for e in remain_head if e != index]
                if keep_num_ >= 96:
                    break

                # print(f'step update del head append:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')



            
            # 计算是否可以全部保留
            # if head_nums - 1 - len(keep_head)-len(del_head) == 0: #这种情况出现在，保留的维度不能很好的均分，然后除了全部保留，和全部删除，只有一个头被用来调整，此时不能调
            #     continue
            # index = remain_head[int(torch.argmax(head_score[remain_head]).cpu().item())]
            # k_now = int((keep_num - len(keep_head)*head_dim-head_dim)/(head_nums-1 - len(keep_head)-len(del_head)))
            # if k_now <=0 :  # 保留的channel数量，不足一个head_dim，这种情况出现在remain head 保留维度比较小的时候，所有remain head的保留维度加一起不足一个head_dim
            #     continue
        
            # cur_score = get_score(keep_head+[index], del_head, score, k_now)
            # if cur_score > sum_score:
            #     # print(f'cur_score is {cur_score}')
            #     keep_head.append(index)
            #     sum_score = cur_score
            #     keep_num_ = k_now
            #     remain_head =  [e for e in remain_head if e != index]
            #     print(f'step update keep head append:  keep_num is {keep_num}, keep_head is {keep_head}, del_head is {del_head}, remain_head is {remain_head}, keep_num_ is {keep_num_}')

    # 修正keep_num_，上面的keep_num_选择比较保守，可能会产生更高的稀疏度，因此在此处按照四舍五入进行修正，
    keep_num_ = int(round(keep_num/len(remain_head)/2)*2)
    logger.info(f'keep haeds are {keep_head}, del head are {del_head},  keep_num form {keep_num_h} to {keep_num_}')
    keep_num_h = keep_num_
    # 先剪枝rope
    keep_head_num = []
    for i in range(head_nums):
        if i in del_head:
            keep_head_num.append(0)
        elif i in keep_head:
            keep_head_num.append(head_dim)
        else:
            keep_head_num.append(keep_num_h)
    logger.info(f'head keep is :{keep_head_num}' )
    rope_Q = ROPE_LM_PCA(qs=q, ks=k, layer=layer, keep_num=keep_head_num, group=groups)


    recon_inpx =[get_recon_out(q, k, v, del_head, rope_Q) for q,k,v in zip(q, k, v)] 




    

    ## 计算修正后的
    H = None
    for idx, X_batch in enumerate(recon_inpx):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double().cuda()
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch
        del X_batch
   
    # 
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).cuda()
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H.cuda().double())
    
    score = X_eig[1]**2@torch.reshape(X_eig[0], shape=(H.shape[1], 1))
    _, sorted_indices = torch.topk(score.view(-1), k=H.shape[0])
    mask_Q = torch.zeros(size=(H.shape[1], keep_num), dtype=torch.double)
    mask_Q[sorted_indices[:keep_num], range(keep_num)] = 1  
    
    h = torch.sum(mask_Q, dim=1)
    head_num = torch.sum(h.view(head_nums, -1), dim=1).to(int)  # 每个头保留的channel数量
    

    # 计算单个头的pca
    Q = torch.zeros_like(H).cuda()
    val = torch.zeros_like(H[0]).cuda()
    if groups ==1:
        for i in range(head_nums):
            X_eig = torch.linalg.eigh(H[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim])
            index = torch.argsort(X_eig[0], descending=True)
            eig_val = X_eig[0][index]
            eigen_vec = X_eig[1][:, index]
            Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = eigen_vec
            val[i*head_dim:i*head_dim+head_dim] = eig_val/torch.sum(eig_val)  # 归一化特征值，用于后续头保留
    else:
        H_patch = None
        for i in range(head_nums):
            if i not in del_head:  # 只对保留的head做压缩，提升精度
                H_patch = H[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] if H_patch is None else H_patch+H[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim]
            if (i+1)%groups ==0:
                if H_patch is not None:  # 如果是None，表示整个group都剪掉了
                    X_eig = torch.linalg.eigh(H_patch)
                    index = torch.argsort(X_eig[0], descending=True)
                    eig_val = X_eig[0][index]
                    eigen_vec = X_eig[1][:, index]
                    H_patch = None
                    for j in range(i//groups*groups,i//groups*groups+groups):
                        Q[j*head_dim:j*head_dim+head_dim, j*head_dim:j*head_dim+head_dim] = eigen_vec
                        val[j*head_dim:j*head_dim+head_dim] = eig_val/torch.sum(eig_val)  # 归一化特征值，用于后续头保留
    
    
    new_Q  = torch.zeros(H.shape).cuda().to(dtype=H.dtype)
    for i in range(head_nums):
       
        if i in del_head:
            continue
        if i in keep_head:
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim]
        else:
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim] = Q[i*head_dim:i*head_dim+head_dim, i*head_dim:i*head_dim+head_dim]
            new_Q[i*head_dim:i*head_dim+head_dim, i*head_dim+keep_num_h:i*head_dim+head_dim] = 0
    new_x = [(x.cuda().double()@new_Q.cuda().double()@new_Q.T.cuda().double()).cpu() for x in recon_inpx]
    Q_t = get_pinverse_svd(new_x, X)
       

    return rope_Q.double().cuda(), new_Q.double().cuda(), new_Q.T.double().cuda()@Q_t.double().cuda() 
