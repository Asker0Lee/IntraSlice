import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
import transformers
import warnings
import math
from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.phi3.modeling_phi3 import Phi3Model, Phi3DecoderLayer, Phi3SdpaAttention, Phi3ForCausalLM,Phi3Attention, Phi3RMSNorm, Cache,apply_rotary_pos_emb,repeat_kv
from transformers.models.phi3.modeling_phi3 import logger, PHI3_INPUTS_DOCSTRING, add_start_docstrings_to_model_forward, BaseModelOutputWithPast, DynamicCache
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
from typing import Any
from tqdm import tqdm
import copy
from ..utils import find_layers
from ..prune_utils_phi3 import Pruner
'''
Pruning adapter for Llama models
'''

class Slice_Phi3SdpaAttention2(Phi3SdpaAttention):
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
            )
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if self.rope_Q is not None:
            self.ropeK =  torch.reshape(self.rope_Q, shape=(bsz, self.num_key_value_heads, -1, self.head_dim, self.head_dim))[:,:,0]     #(1, k_num, 128, 128)
            query_states = query_states@self.rope_Q
            key_states = key_states@self.ropeK

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
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



class Slice_Phi3Attention(Phi3Attention):
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if self.rope_Q is not None:
            self.ropeK =  torch.reshape(self.rope_Q, shape=(bsz, self.num_key_value_heads, -1, self.head_dim, self.head_dim))[:,:,0]     #(1, k_num, 128, 128)
            query_states = query_states@self.rope_Q
            key_states = key_states@self.ropeK

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


@add_start_docstrings_to_model_forward(PHI3_INPUTS_DOCSTRING)
def compressed_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
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

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    use_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        use_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

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

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class CompressedPhi3DecoderLayer(Phi3DecoderLayer):
    """
    This class simulates the Phi3DecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py#L817)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # self.self_attn = Slice_Phi3SdpaAttention2(config=config, layer_idx=layer_idx)
        self.self_attn = Slice_Phi3Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



'''
load model
replace model

获得激活值
剪枝函数


'''

def get_phi3(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    kwargs = {'attn_implementation': "eager"}
    model = Phi3ForCausalLM.from_pretrained(model, torch_dtype=torch.float16, **kwargs) #,device_map="auto"
    model.seqlen = 2048
    return model

class Phi3ModelAdapter():
    def __init__(self, model: Phi3ForCausalLM) -> None:
        super().__init__()
        self._model: Phi3ForCausalLM = model
        

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    def replace_forward(self):
        self._model.model.forward = compressed_forward.__get__(self._model.model, Phi3Model)
        print('forward replace done!')

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def original_layer_type(self) -> type:
        return Phi3DecoderLayer

    @property
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of strings specifying the class names of modules that should not be split.
        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
        """
        return [self.original_layer_type.__name__, self.compressed_layer_type.__name__]
    
    @property
    def compressed_layer_type(self) -> type:
        return CompressedPhi3DecoderLayer


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
                
                
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                #如果是down_proj，获取up的输出和gate的输出
                if 'mlp.down_proj' in name:
                    def hook_up_gate(name):
                        def tmp(_, args: tuple, _output: Any) -> None:
                            out = _output.clone()  # Position in RMSN.forward args
                            gate, up_states = out.chunk(2, dim=-1)
                            pruner[name].mlp_activation['up_out'].append(up_states.cpu())
                            pruner[name].mlp_activation['gate_out'].append(gate.cpu())
                        return tmp
                    handles.append(block_layers['mlp.gate_up_proj'].register_forward_hook(hook_up_gate(name)))
                    

                if 'self_attn.o_proj' in name:
                    def hook_qk(name):
                        def get_qk_hook(m,args,  kwargs, result):
                            assert isinstance(kwargs, dict)
                            
                            hidden_states = kwargs["hidden_states"]
                            position_ids = kwargs["position_ids"]

                            bsz, q_len, _ = hidden_states.size()

                            qkv = m.qkv_proj(hidden_states)
                            query_pos = m.num_heads * m.head_dim
                            q = qkv[..., :query_pos].view(bsz, q_len, m.num_heads, m.head_dim).transpose(1, 2)
                            k = qkv[..., query_pos : query_pos + m.num_key_value_heads * m.head_dim].view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)
                            v = qkv[..., query_pos + m.num_key_value_heads * m.head_dim :].view(bsz, q_len, m.num_key_value_heads, m.head_dim).transpose(1, 2)

                            kv_seq_len = k.shape[-2]
                            cos, sin = m.rotary_emb(v, position_ids, seq_len=kv_seq_len)
                            cos, sin = m.rotary_emb(v, position_ids)
                            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                            
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