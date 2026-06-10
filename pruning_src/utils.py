import datetime
import gc
import inspect
import logging
import time
import pathlib
from typing import TypeVar
import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM,OPTForCausalLM,BloomForCausalLM, Phi3ForCausalLM
from torch.utils.data import DataLoader
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.hooks import remove_hook_from_module
def configure_logging(
        args,
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_dir: str = 'log',
        level: int = logging.INFO,
) -> None:
    
    logger = logging.getLogger('InterSlice')
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_to_console:
        handler = logging.StreamHandler()
        # handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_to_file:
        model_name = args.model.split('/')[-1]
        log_dir = os.path.join(log_dir, f'{model_name}-{args.sparsity}-{args.dataset}-{args.nsamples}')
        os.makedirs(log_dir, exist_ok=True)  
        log_filepath =os.path.join(log_dir,f'iterpca={args.iterpca}_{args.global_layer_rate}_bias={args.global_bias}_frac={args.global_frac}_{datetime.datetime.now():log_%Y-%m-%d-%H-%M-%S}.log')
       
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        # file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)04d\t%(levelname)s\t%(name)s\t%(message)s', datefmt='%Y-%m-%dT%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger




def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    


def distribute_model(model_adapter) -> None:
    """Distribute the model across available GPUs."""
    model = model_adapter.model
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model_adapter.no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model_adapter.no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    # Run GC and cleanup GPU memory
    cleanup_memory()

def consolidate_model_to_single_gpu(model, target_device='cuda:0'):
    """将分布式模型整合到单GPU"""
     
    try:
        # 移除所有加速器钩子
        for name, module in model.named_modules():
            remove_hook_from_module(module, recurse=True)
        
        # 清除设备映射
        if hasattr(model, 'hf_device_map'):
            delattr(model, 'hf_device_map')
        
        # 收集所有参数到目标设备
        model = model.to(target_device)
        
        # 清理内存
        cleanup_memory()
        
        logging.info(f"Model consolidated to {target_device}")
        return model
        
    except Exception as e:
        logging.error(f"Failed to consolidate model: {e}")
        raise e

def unwarp_distribute_model(model_adapter) -> None:
    
    model = model_adapter.model
    device_map = {"":0}
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True
    )

    # Run GC and cleanup GPU memory
    cleanup_memory()

@torch.no_grad()
def evaluate_ppl(model, pad_token_id, testloader) -> float:
    """
   
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in testloader:
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, 'cuda')
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()





def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        logging.debug(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


T = TypeVar('T')


def map_tensors(obj: T, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> T:
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





    


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)



def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_model_properties(model):
    if isinstance(model, OPTForCausalLM):
        num_hidden_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        ffn_dim = model.config.ffn_dim
    elif isinstance(model, BloomForCausalLM):
        num_hidden_layers = model.config.n_layer
        num_heads = model.config.n_head
        ffn_dim = model.config.hidden_size * 4
    elif isinstance(model,LlamaForCausalLM):
        num_hidden_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        ffn_dim = model.config.intermediate_size
    elif isinstance(model,Phi3ForCausalLM):
        num_hidden_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        ffn_dim = model.config.intermediate_size

    hidden_size = model.config.hidden_size
    head_size = int(hidden_size / num_heads)

    return num_hidden_layers,num_heads,ffn_dim,hidden_size,head_size


