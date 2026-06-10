import os
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] ='2'
from typing import Optional
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, DynamicCache
from modules.sobp_module import SoBPLlamaDecoderLayer
from modules.llm_surgeon_module import LLM_Surgeon_LlamaDecoderLayer
from modules.disp_llm_module import DISP_LlamaDecoderLayer
from modules.IntraSlice_module import IntraSliceLlamaDecoderLayer
from modules.llama_baseline import BaseLlamaDecoderLayer
import time

def count_parameters(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())

def create_kv_cache(model:LlamaDecoderLayer, batch, length, device):
    kv_head_num = model.self_attn.num_key_value_heads
    kv_head_dim = model.self_attn.head_dim
    dtype = model.self_attn.config.torch_dtype
    # [B, kv_head_num, length, kv_head_dim]
    key_cache = torch.randn(size=(batch, kv_head_num, length, kv_head_dim)).to(device=device, dtype=dtype)
    value_cache = torch.randn(size=(batch, kv_head_num, length, kv_head_dim)).to(device=device, dtype=dtype)
    past_key_values = DynamicCache.from_legacy_cache([(key_cache, value_cache)])
    return past_key_values


def test_prefill_speed(sparsity, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    batch_size = 4
    seq_len = 4096
    hidden_size = config.hidden_size
    # 位置编码
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L)

    # 构造输入
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask = attention_mask[:, None, None, :]

    # 实例化 DecoderLayer
    baseline_layer = LlamaDecoderLayer(config, 0).to(device)
    baseline_param = count_parameters(baseline_layer)
    sobp_layer = SoBPLlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    sobp_param = count_parameters(sobp_layer)
    llmsurgeon_layer = LLM_Surgeon_LlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    llmsurgeon_param = count_parameters(llmsurgeon_layer)
    disp_layer = DISP_LlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    disp_param = count_parameters(disp_layer)
    intraslice_layer = IntraSliceLlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    intraslice_param = count_parameters(intraslice_layer)

    


    # warm-up
    for _ in range(10):
        _ = baseline_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
        _ = sobp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
        _ = llmsurgeon_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
        _ = disp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
        _ = intraslice_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)

    # baseline 测试
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(50):
        _ = baseline_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    baseline_time = time.time() - t1

    # sobp 测试
    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(50):
        _ = sobp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    sobp_time = time.time() - t2

    # llm_surgeon 测试
    torch.cuda.synchronize()
    t3 = time.time()
    for _ in range(50):
        _ = llmsurgeon_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    llm_surgeon_time = time.time() - t3

    # disp-llm 测试
    torch.cuda.synchronize()
    t4 = time.time()
    for _ in range(50):
        _ = disp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    disp_time = time.time() - t4

    # intraslice 测试
    torch.cuda.synchronize()
    t5 = time.time()
    for _ in range(50):
        _ = intraslice_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    intraslice_time = time.time() - t5

    # 输出
    print(f"Baseline (LlamaDecoderLayer) time: {baseline_time:.4f} s; \n  count_parameters:{baseline_param}")
    print('---------------------------------------------------------------')
    print(f"SoBP (SoBPLlamaDecoderLayer) time: {sobp_time:.4f} s; \n  count_parameters:{sobp_param}")
    print(f"  Speedup ratio: {baseline_time / sobp_time:.2f}x; \n  Prune ratio:{1-sobp_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"llm-surgeon (LLM_Surgeon_LlamaDecoderLayer) time: {llm_surgeon_time:.4f} s;\n count_parameters:{llmsurgeon_param}")
    print(f"  Speedup ratio: {baseline_time / llm_surgeon_time:.2f}x;\n  Prune ratio:{1-llmsurgeon_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"disp-llm (DISP_LlamaDecoderLayer) time: {disp_time:.4f} s;\n count_parameters:{disp_param}")
    print(f"  Speedup ratio: {baseline_time / disp_time:.2f}x;\n  Prune ratio:{1-disp_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"intraslice (IntraSliceLlamaDecoderLayer) time: {intraslice_time:.4f} s;\n count_parameters:{intraslice_param}")
    print(f"  Speedup ratio: {baseline_time / intraslice_time:.2f}x;\n  Prune ratio:{1-intraslice_param/baseline_param:.2f}")



def test_generate_speed(sparsity, kv_length, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    batch_size = 4
    seq_len = 1
    hidden_size = config.hidden_size
    # 位置编码
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L)

    # 构造输入
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    attention_mask = torch.ones(batch_size, seq_len+kv_length, device=device)
    attention_mask = attention_mask[:, None, None, :]

    # 实例化两个 DecoderLayer
   
    baseline_layer = BaseLlamaDecoderLayer(config, 0).to(device)
    baseline_param = count_parameters(baseline_layer)
    sobp_layer = SoBPLlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    sobp_param = count_parameters(sobp_layer)
    llmsurgeon_layer = LLM_Surgeon_LlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    llmsurgeon_param = count_parameters(llmsurgeon_layer)
    disp_layer = DISP_LlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    disp_param = count_parameters(disp_layer)
    intraslice_layer = IntraSliceLlamaDecoderLayer(config, 0, sparsity=sparsity).to(device)
    intraslice_param = count_parameters(intraslice_layer)

    # create kv cache
    baseline_kvcache = create_kv_cache(baseline_layer, batch=batch_size, length=kv_length, device=device)
    sobp_kvcache = create_kv_cache(sobp_layer, batch=batch_size, length=kv_length, device=device)
    llmsurgeon_kvcache = create_kv_cache(llmsurgeon_layer, batch=batch_size, length=kv_length, device=device)
    disp_kvcache = create_kv_cache(disp_layer, batch=batch_size, length=kv_length, device=device)
    intraslice_kvcache = create_kv_cache(intraslice_layer, batch=batch_size, length=kv_length, device=device)

    # warm-up
    for _ in range(10):
        _ = intraslice_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=intraslice_kvcache)
        _ = baseline_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=baseline_kvcache)
        _ = sobp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=sobp_kvcache)
        _ = llmsurgeon_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=llmsurgeon_kvcache)
        _ = disp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=disp_kvcache)
        

    # baseline 测试
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(50):
        _ = baseline_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=baseline_kvcache)
    torch.cuda.synchronize()
    baseline_time = time.time() - t1

    # sobp 测试
    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(50):
        _ = sobp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=sobp_kvcache)
    torch.cuda.synchronize()
    sobp_time = time.time() - t2

    # llm_surgeon 测试
    torch.cuda.synchronize()
    t3 = time.time()
    for _ in range(50):
        _ = llmsurgeon_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=llmsurgeon_kvcache)
    torch.cuda.synchronize()
    llm_surgeon_time = time.time() - t3

    # disp-llm 测试
    torch.cuda.synchronize()
    t4 = time.time()
    for _ in range(50):
        _ = disp_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=disp_kvcache)
    torch.cuda.synchronize()
    disp_time = time.time() - t4

    # intraslice 测试
    torch.cuda.synchronize()
    t5 = time.time()
    for _ in range(50):
        _ = intraslice_layer(input_tensor, attention_mask=attention_mask, position_ids=position_ids, past_key_value=intraslice_kvcache)
    torch.cuda.synchronize()
    intraslice_time = time.time() - t5

    # 输出
    print(f"Baseline (LlamaDecoderLayer) time: {baseline_time:.4f} s; \n  count_parameters:{baseline_param}")
    print('---------------------------------------------------------------')
    print(f"SoBP (SoBPLlamaDecoderLayer) time: {sobp_time:.4f} s; \n  count_parameters:{sobp_param}")
    print(f"  Speedup ratio: {baseline_time / sobp_time:.2f}x; \n  Prune ratio:{sobp_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"llm-surgeon (LLM_Surgeon_LlamaDecoderLayer) time: {llm_surgeon_time:.4f} s;\n count_parameters:{llmsurgeon_param}")
    print(f"  Speedup ratio: {baseline_time / llm_surgeon_time:.2f}x;\n  Prune ratio:{llmsurgeon_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"disp-llm (DISP_LlamaDecoderLayer) time: {disp_time:.4f} s;\n count_parameters:{disp_param}")
    print(f"  Speedup ratio: {baseline_time / disp_time:.2f}x;\n  Prune ratio:{disp_param/baseline_param:.2f}")
    print('---------------------------------------------------------------')
    print(f"intraslice (IntraSliceLlamaDecoderLayer) time: {intraslice_time:.4f} s;\n count_parameters:{intraslice_param}")
    print(f"  Speedup ratio: {baseline_time / intraslice_time:.2f}x;\n  Prune ratio:{intraslice_param/baseline_param:.2f}")

if __name__ == '__main__':
    # generate
    model_name = "meta-llama/Llama-2-13b-hf"
    config = AutoConfig.from_pretrained(model_name)
    print(f'========================= {model_name} speed test ==========================')
    for each_sparsity in [0.5]:
        for length in [1024, 2048, 4096, 1024*8, 1024*16]:
            print(f'+++++++++++++ sprsity is {each_sparsity}, kv_length={length} ++++++++++++++++++')
            test_generate_speed(each_sparsity, length, config )

    # prefill
    model_name = "meta-llama/Llama-2-13b-hf"
    config = AutoConfig.from_pretrained(model_name)
    print(f'========================= {model_name} speed test ==========================')
    for each_sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f'+++++++++++++ sprsity is {each_sparsity} ++++++++++++++++++')
        test_prefill_speed(each_sparsity, config)

    # model_name = "meta-llama/Meta-Llama-3-8B"
    # config = AutoConfig.from_pretrained(model_name)
    # print(f'========================= {model_name} speed test ==========================')
    # for each_sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     print(f'+++++++++++++ sprsity is {each_sparsity} ++++++++++++++++++')
    #     test_speed(each_sparsity, config)
    # model_name = "meta-llama/Llama-2-70b-hf"
    # config = AutoConfig.from_pretrained(model_name)
    # print(f'========================= {model_name} speed test ==========================')
    # for each_sparsity in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     print(f'+++++++++++++ sprsity is {each_sparsity} ++++++++++++++++++')
    #     test_speed(each_sparsity, config)

    


