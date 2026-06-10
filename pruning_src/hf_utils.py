# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Any
import logging
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter
from typing import Callable, Iterable, Optional, TypeVar
from transformers import AutoTokenizer, LlamaForCausalLM,OPTForCausalLM,BloomForCausalLM
from .adapters.llama_adapter import LlamaModelAdapter, get_llama
from .adapters.phi3_adapter import Phi3ModelAdapter, get_phi3

def do_not_initialize(func):
    """
    A decorator that prevents initialization of torch.nn modules.
    """

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(model_name: str):
    """
   
    """
    if 'opt' in model_name.lower():
        model = get_opt(model_name)
    elif 'phi-3' in model_name.lower():
        
        model = get_phi3(model_name)
        model.config.torch_dtype = torch.float16
        model = Phi3ModelAdapter(model)
    elif 'llama' in model_name.lower():
        model = get_llama(model_name)
        model.config.torch_dtype = torch.float16
        model = LlamaModelAdapter(model)
    else:
        raise ValueError("not supported model")
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    logging.info(f'Model loading done.')
    model.replace_forward()  # 替换forward
    replace_layers(model)
    return model, tokenizer





def replace_layers(model_adapter, verbose: bool = True) -> None:
    if verbose:
        logging.info("Replacing layers")

    replace_modules(
        model_adapter.model,
        model_adapter.original_layer_type,
        model_adapter.convert_layer_to_compressed,
        replace_layers=True,
    )

    if verbose:
        logging.info("Replacing layers done")


AnyModule = TypeVar("AnyModule", bound=Module)


def replace_modules(
    root: Module,
    type_to_replace: type[AnyModule],
    new_module_factory: Callable[
        [AnyModule, Optional[int]],
        Module,
    ],
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)














# @do_not_initialize
# def load_sliced_model(
#     model_name: str,
#     sliced_model_path: str,
#     *,
#     token: str | None = None,
#     lora_config: Any = None,
#     sparsity: float | None = None,
#     round_interval: int | None = 1,
# ) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
#     """
#     Load the sliced model and the tokenizer from the given path. If lora_config: peft.LoraConfig is supplied
#     as an arg then this function will return a PEFT model (post-slicing finetuned model). Despite being declared as
#     "Any", lora_config is supposed to have the type peft.LoraConfig. It has type "Any" in the function's signature,
#     so that it would be possible to use it without taking a dependency on peft, when one is not required.
#     The corresponding model adapter class must be imported before calling this method.
#     """
#     my_model_suffix = pathlib.Path(model_name).name
#     my_sliced_model_name = f"{my_model_suffix}_{sparsity}.pt"
#     my_sliced_model_config = f"{my_model_suffix}_{sparsity}.json"

#     model_adapter, tokenizer = get_model_and_tokenizer(
#         model_name,
#         model_path=sliced_model_path,
#         uninitialized=True,
#         token=token,
#     )
#     replace_layers(model_adapter)
#     fuse_modules(model_adapter)

#     hidden_size = model_adapter.hidden_size
#     for layer_adapter in model_adapter.get_layers():
#         if not model_adapter.parallel_blocks:
#             layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
#                 torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
#             )
#         layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
#             torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
#         )

#     config_path = pathlib.Path(sliced_model_path) / my_sliced_model_config

#     if config_path.exists():
#         model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

#     if model_adapter.slicing_conf is None:
#         # assume the model was sliced with the const sparsity specified in the arguments to this method
#         new_embedding_dimension = int((1 - sparsity) * hidden_size)
#         new_embedding_dimension -= new_embedding_dimension % round_interval
#         config = SlicingConfig()
#         config.const_dimension = new_embedding_dimension
#         model_adapter.slicing_conf = config

#     slice_rotated_model(model_adapter)

#     if lora_config:
#         from peft import get_peft_model

#         model_adapter.model = get_peft_model(model_adapter.model, lora_config)

#     logging.info(f"Loading sliced model weights from {sliced_model_path}")
#     model_adapter.model.load_state_dict(
#         torch.load(str(pathlib.Path(sliced_model_path) / my_sliced_model_name), map_location="cpu")
#     )
#     model_adapter.model.eval()

#     return model_adapter, tokenizer
