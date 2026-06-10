import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
import transformers
import math
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import logger, LlamaConfig, LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM,  LlamaSdpaAttention, LlamaMLP, Cache,apply_rotary_pos_emb,repeat_kv, rotate_half
from transformers.models.llama.modeling_llama import DynamicCache, LLAMA_INPUTS_DOCSTRING,BaseModelOutputWithPast, add_start_docstrings_to_model_forward, LlamaModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
from typing import Any
from tqdm import tqdm
import copy
from ..utils import find_layers
from ..prune_utils_llama import Pruner
'''
Pruning adapter for Llama models
'''



class Slice_LlamaSdpaAttention2(LlamaSdpaAttention):
    def __init__(self, config, layer_idx = None):
        super().__init__(config, layer_idx)
        self.register_buffer('rope_Q', None)
        

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
         **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.rope_Q is not None:
            self.ropeK =  torch.reshape(self.rope_Q, shape=(bsz, self.num_key_value_heads, -1, self.head_dim, self.head_dim))[:,:,0]     #(1, k_num, 128, 128)
            query_states = query_states@self.rope_Q
            key_states = key_states@self.ropeK

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            if self.head_prune_config is not None:
                value_states_full = value_states_full.contiguous()
                value_states_prune = value_states_full.contiguous()
            else:
                value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False
        
           
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
       
        
        return attn_output, None, past_key_value


class Slice_LlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.register_buffer('rope_Q', None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.rope_Q is not None:
            self.ropeK =  torch.reshape(self.rope_Q, shape=(bsz, self.num_key_value_heads, -1, self.head_dim, self.head_dim))[:,:,0]     #(1, k_num, 128, 128)
            query_states = query_states@self.rope_Q
            key_states = key_states@self.ropeK

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Slice_LlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        down_inputs = self.act_fn(self.gate_proj(x))*self.up_proj(x)
        down_proj = self.down_proj(down_inputs)
        return down_proj



class CompressedLlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.Q2 = None
        self.self_attn = Slice_LlamaSdpaAttention2(config=config, layer_idx=layer_idx)
        # self.self_attn = Slice_LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = Slice_LlamaMLP(config)
        

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def compressed_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # if self.gradient_checkpointing and self.training:
            #     layer_outputs = self._gradient_checkpointing_func(
            #         decoder_layer.__call__,
            #         hidden_states,
            #         causal_mask,
            #         position_ids,
            #         past_key_values,
            #         output_attentions,
            #         use_cache,
            #         cache_position,
            #     )
        from torch.utils.checkpoint import checkpoint
        if self.gradient_checkpointing:
                layer_outputs = checkpoint(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16) #,device_map="auto"
    model.seqlen = 2048
    return model

class LlamaModelAdapter():
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    def replace_forward(self):
        # self._model.model.forward = compressed_forward.__get__(self._model.model, LlamaModel)
        print('forward replace done!')

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer

    @property
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of strings specifying the class names of modules that should not be split.
        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
        """
        return [self.original_layer_type.__name__, self.compressed_layer_type.__name__]
    
    @property
    def compressed_layer_type(self) -> type:
        return CompressedLlamaDecoderLayer


    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer


    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Llama-2 and Llama-3 don't have a pad tokens by default
        tokenizer.pad_token = tokenizer.eos_token
        self.config.pad_token_id = tokenizer.pad_token_id
        
    @torch.no_grad()
    def prune_model(self, args, dataloader, global_prune_ratio, dev, logger):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.model.layers
        self.layer_num = len(layers)

        self.model.model.embed_tokens = self.model.model.embed_tokens.to(dev)
        self.model.model.norm = self.model.model.norm.to(dev)
        layers[0] = layers[0].to(dev)
        self.model.model.rotary_emb = self.model.model.rotary_emb.to(dev)

        # dtype = next(iter(self.model.parameters())).dtype

        inps = []
        cache = {'attention_mask': None, 'position_ids': None}
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps.append(inp.detach().clone().cpu())
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                self.model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        self.model.model.norm = self.model.model.norm.cpu()
        self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        self.model.cpu()
        torch.cuda.empty_cache()

        # update_preceding = [False for i in range(len(layers))]

        # if args.error_compensation and False:
        #     mid = len(layers)//2 + 1
        #     update_preceding[-mid] = True
        #     update_preceding[-2] = True

        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        num_batches = len(inps)
        ori_inps = copy.deepcopy(inps)
        logger.info('#################### start pruning ###################')
        for i in tqdm(range(len(layers)),desc='pruning...'):
            layer = layers[i].to(dev)
            logger.info(f'######## pruning layer {i} #########')
            ori_ffn_residual = []
            ffn_residual = []
            ori_up_proj_inp = []

            for j in range(num_batches):
                outputs = layer(ori_inps[j].to(dev), attention_mask=attention_mask, position_ids=position_ids,\
                                out_ffn_residual=True, out_up_proj_inp=True)
                ori_inps[j] = outputs[0].cpu()
                # ori_ffn_residual.append(outputs[1].cpu())
                # ori_up_proj_inp.append(outputs[2].cpu())

            block_layers = find_layers(layer)
            sequential = ['self_attn.o_proj', 'mlp.down_proj']
            for name in sequential:
                subset = {name: block_layers[name]}
                pruner = {}
                pruner[name] = Pruner(subset[name], module_name=f'{name}', layers=layer, logger=logger)

                def add_batch(name,preceding_inps=None):
                    def tmp(_, inp, out):
                        if 'mlp.down_proj' in name:
                            pruner[name].mlp_activation['down_in'].append(inp[0].data.cpu())
                        if 'self_attn.o_proj' in name:
                            pruner[name].attn_activation['out_in'].append(inp[0].data.cpu())
                        if preceding_inps is None:
                            pruner[name].add_batch(inp[0].data)
                        else:
                            pruner[name].preceding_add_batch(inp[0].data,preceding_inps)
                    return tmp

                handles = []
                
                if 'mlp.up_proj' in name:
                    if update_preceding[i]:
                        handles.append(subset[name].register_forward_hook(add_batch(name,ori_up_proj_inp)))
                        # pruner[name].final_layer_norm = layer.post_attention_layernorm
                    else:
                        continue
                else:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                #如果是down_proj，获取up的输出和gate的输出
                if 'mlp.down_proj' in name:
                    def hook_up(name):
                        def tmp(_, args: tuple, _output: Any) -> None:
                            out = _output.clone()  # Position in RMSN.forward args
                            pruner[name].mlp_activation['up_out'].append(out.cpu())
                        return tmp

                    
                    def hook_gate(name) -> None:
                        def tmp(_, args: tuple, _output: Any) -> None:
                            out = _output.clone()  # Position in RMSN.forward args
                            pruner[name].mlp_activation['gate_out'].append(out.cpu())
                        return tmp
                        
                    handles.append(block_layers['mlp.up_proj'].register_forward_hook(hook_up(name)))
                    handles.append(block_layers['mlp.gate_proj'].register_forward_hook(hook_gate(name)))

                if 'self_attn.o_proj' in name:
                    def hook_qk(name):
                        def get_qk_hook(m,args,  kwargs, result):
                            assert isinstance(kwargs, dict)
                            
                            hidden_states = kwargs["hidden_states"]
                            position_ids = kwargs["position_ids"]

                            bsz, q_len, _ = hidden_states.size()

                            q = m.q_proj(hidden_states).view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
                            k = m.k_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
                            v = m.v_proj(hidden_states).view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
                            kv_seq_len = k.shape[-2]
                        
                            cos, sin = m.rotary_emb(v, position_ids)
                            q, k = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(q, k, cos, sin)
                            
                            k = repeat_kv(k, m.num_key_value_groups)
                            v = repeat_kv(v, m.num_key_value_groups)
                            
                            pruner[name].attn_activation['q_out'].append(q.cpu())
                            pruner[name].attn_activation['k_out'].append(k.cpu())
                            pruner[name].attn_activation['v_out'].append(v.cpu())
                        return get_qk_hook
                        
                    handles.append(layer.self_attn.register_forward_hook(hook_qk(name), with_kwargs=True))
                    

                for j in range(num_batches):
                    if 'mlp.up_proj' in name:
                        outputs = layer(inps[j].to(dev), attention_mask=attention_mask,\
                                        position_ids=position_ids,out_ffn_residual=True)
                        ffn_residual.append(outputs[1].cpu())
                    else:
                        layer(inps[j].to(dev), attention_mask=attention_mask, position_ids=position_ids)

                for h in handles:
                    h.remove()

                if 'self_attn.o_proj' in name:  # head
                    num_consecutive = self.model.config.hidden_size // self.model.config.num_attention_heads
                   
                    
                    prune_ratio = global_prune_ratio[i][0].item()
                    # pruner[name].get_mask_weight_adjust(prune_ratio, num_consecutive, valid_head_mask,percdamp=args.percdamp)
                    pruner[name].get_mask_weight_adjust_mha2(prune_ratio)

                elif 'mlp.down_proj' in name:  # ffn
                    prune_ratio = global_prune_ratio[i][1].item()
                    if args.iterpca:
                        if i <2 or self.layer_num-i < 3:
                            pruner[name].get_mask_weight_adjust_lm(prune_ratio)
                        else:
                            pruner[name].get_mask_weight_adjust_iterpca_lm(prune_ratio)
                    else:
                        pruner[name].get_mask_weight_adjust_lm(prune_ratio)
                    # pruner[name].get_mask_weight_adjust_pca(prune_ratio, num_consecutive, valid_neuron_mask,percdamp=args.percdamp)

            outs=  []
            for j in range(args.nsamples):
                out = layer(inps[j].to(dev),attention_mask=attention_mask, position_ids=position_ids)[0]
                out = torch.nan_to_num(out)
                outs.append(out.cpu())


            layers[i] = layer.cpu()
            del layer
            del pruner
            torch.cuda.empty_cache()
            inps = outs

        self.model.config.use_cache = use_cache
        del inps
        del outs