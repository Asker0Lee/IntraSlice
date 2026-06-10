import torch
import torch.nn as nn
import time
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3DecoderLayer

no_split_classes = ["LlamaDecoderLayer", "InternLM2DecoderLayer"]

import os
import gc

def register_mask(module, mask):
    def hook(_, inputs):
        nonlocal mask
        mask = mask.to(inputs[0].device)
        return (inputs[0] * mask,)

    handle = module.register_forward_pre_hook(hook)
    return handle

def register_H(module, H):
    def hook(_, inputs):
        # nonlocal H
        with torch.no_grad():
            X_batch = inputs[0].clone().float()
            H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
            H.add_(H_batch)
            del X_batch, H_batch
            torch.cuda.empty_cache()
            gc.collect()

    handle = module.register_forward_pre_hook(hook)
    return handle

def apply_mask(model, neuron_mask, head_mask):
    handles = []
    num_hidden_layers = neuron_mask.shape[0]
    for layer_idx in range(num_hidden_layers): 
        decoder_block = model.model.layers[layer_idx]
        if isinstance(decoder_block, LlamaDecoderLayer):
            # register attn mask
            head_dim = decoder_block.self_attn.head_dim
            a_mask = head_mask[layer_idx] # （head_nums）
          
            # a_mask = torch.repeat_interleave(a_mask, head_dim) # 间隔重复
            o_ffn = decoder_block.self_attn.o_proj

            handle = register_mask(o_ffn, a_mask)
            handles.append(handle)

            # register mlp mask
            n_mask = neuron_mask[layer_idx] # （head_nums）
         
            d_ffn = decoder_block.mlp.down_proj
            handle = register_mask(d_ffn, n_mask)
            handles.append(handle)
        elif isinstance(decoder_block, Phi3DecoderLayer):
            head_dim = decoder_block.self_attn.head_dim
            a_mask = head_mask[layer_idx] # （head_nums）
          
            # a_mask = torch.repeat_interleave(a_mask, head_dim) # 间隔重复
            o_ffn = decoder_block.self_attn.o_proj

            handle = register_mask(o_ffn, a_mask)
            handles.append(handle)

            # register mlp mask
            n_mask = neuron_mask[layer_idx] # （head_nums）
         
            d_ffn = decoder_block.mlp.down_proj
            handle = register_mask(d_ffn, n_mask)
            handles.append(handle)

    return handles



def apply_H(model, activate_head_H, activate_neuron_H):
    handles = []
    num_hidden_layers = activate_head_H.shape[0]
    for layer_idx in range(num_hidden_layers): 
        decoder_block = model.model.layers[layer_idx]
        if isinstance(decoder_block, LlamaDecoderLayer):
            # register attn mask
            head_dim = decoder_block.self_attn.head_dim
            
            a_h = activate_head_H[layer_idx]  # 获取激活的hessian矩阵
            # a_mask = torch.repeat_interleave(a_mask, head_dim) # 间隔重复
            o_ffn = decoder_block.self_attn.o_proj

            handle = register_H(o_ffn, a_h)
            handles.append(handle)

            # register mlp mask
            n_h= activate_neuron_H[layer_idx]
            d_ffn = decoder_block.mlp.down_proj
            handle = register_H(d_ffn, n_h)
            handles.append(handle)
        elif isinstance(decoder_block, Phi3DecoderLayer):
             # register attn mask
            head_dim = decoder_block.self_attn.head_dim
            
            a_h = activate_head_H[layer_idx]  # 获取激活的hessian矩阵
            # a_mask = torch.repeat_interleave(a_mask, head_dim) # 间隔重复
            o_ffn = decoder_block.self_attn.o_proj

            handle = register_H(o_ffn, a_h)
            handles.append(handle)

            # register mlp mask
            n_h= activate_neuron_H[layer_idx]
            d_ffn = decoder_block.mlp.down_proj
            handle = register_H(d_ffn, n_h)
            handles.append(handle)

    return handles


def collect_mask_grads(model, head_mask, neuron_mask , dataloader, save_path):
    
    model.model.gradient_checkpointing = True
    for param in model.parameters():
        param.requires_grad_(False)

    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    handles = apply_mask(model, neuron_mask, head_mask)

    model.eval()
    model.enable_input_require_grads()
    head_grads = []
    neuron_grads = []
    start = time.time()
    loss_fct = nn.CrossEntropyLoss()
    pt_index = 0
    nlls = []
    nlls2 = []
    for index, batch in enumerate(dataloader):
        data = {'input_ids':batch[0], 'labels':batch[0]}
        data = map_tensors(data, 'cuda')
        # batch =batch[0].to(model.device)
        outputs = model(**data)
        torch.cuda.empty_cache()
        loss = outputs.loss
        loss.backward()

        # jisuan ppl
        # shift_logits = outputs.logits[:, :-1, :].contiguous()

        # # shift_label:[1,2047]
        # shift_labels = data['input_ids'][:, 1:]

        # # shift_logits.view(-1, shift_logits.size(-1)): [2047,50272]  shift_labels.view(-1):[2047,]
        # shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        # shift_labels = shift_labels.view(-1).to('cuda')
        # loss2 = loss_fct(shift_logits, shift_labels)
        # loss2.backward()
        # neg_log_likelihood = loss2.float() * 2048
        # nlls.append(neg_log_likelihood)
        nlls2.append(loss.detach().float().cpu()*2048)


        if (index+1)%1 == 0:
            head_grads.append(head_mask.grad.detach().cpu())
            head_mask.grad = None

            neuron_grads.append(neuron_mask.grad.detach().cpu())
            neuron_mask.grad = None
            gc.collect()
            torch.cuda.empty_cache()
        print(f'当前第{index}batch，时间为{time.time()-start}')
    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)
    neuron_mask.requires_grad_(False)
    head_grads_tensor  = torch.stack(head_grads, dim=0)
    neuron_grads_tensor  = torch.stack(neuron_grads, dim=0)

    # ppl = torch.exp(torch.stack(nlls).sum() / (128 * 2048))
    # print(round(ppl.item(), 4), 'sobp')
    ppl = torch.exp(torch.stack(nlls2).sum() / (128 * 2048))
    model.model.gradient_checkpointing = False
    model.config.use_cache = use_cache
    print(round(ppl.item(), 4), 'mine')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(head_grads_tensor , os.path.join(save_path, f'head_grads.pt'))
    torch.save(neuron_grads_tensor , os.path.join(save_path, f'neuron_grads.pt'))

    

@torch.no_grad()
def collect_activation_H(model, activate_head_H, activate_neuron_H, dataloader, save_path):
   

    handles = apply_H(model, activate_head_H, activate_neuron_H)

    model.eval()
   
    start = time.time()
    pt_index = 0
    for index, batch in enumerate(dataloader):
        data = {'input_ids':batch[0], 'labels':batch[1]}
        data = map_tensors(data, 'cuda')
        outputs = model(**data)
        torch.cuda.empty_cache()

        print(f'当前第{index}batch，时间为{time.time()-start}')
    for handle in handles:
        handle.remove()
   
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(activate_head_H, os.path.join(save_path, f'activate_head_H.pt'))
    torch.save(activate_neuron_H, os.path.join(save_path, f'activate_neuron_H.pt'))
    



@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info

def map_tensors(obj, device: torch.device | str | None = None, dtype: torch.dtype | None = None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj