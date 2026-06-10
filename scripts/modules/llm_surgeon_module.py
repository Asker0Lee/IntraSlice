from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import logger, LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM,  LlamaAttention, LlamaMLP, Cache,apply_rotary_pos_emb,repeat_kv, rotate_half
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
import transformers
import math
# for llama2-7b and llama2-13b speed test

class llm_surgeon_LlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx = None):
        
        super().__init__(config, layer_idx)

        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
     
     
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

        if past_key_value is not None:  # stop update
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=-2), torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=-2)

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


class llm_surgeon_LlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        down_inputs = self.act_fn(self.gate_proj(x))*self.up_proj(x)
        down_proj = self.down_proj(down_inputs)
        return down_proj



class LLM_Surgeon_LlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, config, layer_idx, sparsity):
        config._attn_implementation = 'eager'
        super().__init__(config, layer_idx)
        self.self_attn = llm_surgeon_LlamaAttention(config, layer_idx)
        self.compressed_dim = self.self_attn.hidden_size
        self.sparse_model(sparsity)
        
    @property
    def per_head_param(self):
        hidden_size = self.self_attn.hidden_size
        attention_head_size = self.self_attn.head_dim
        per_head_qkv = hidden_size*attention_head_size + 2/self.self_attn.num_key_value_groups*hidden_size*attention_head_size
        per_head_output = hidden_size*attention_head_size
        param = per_head_qkv+per_head_output
        return param
    
    @property
    def per_neuron_param(self):
        return self.mlp.hidden_size*3


    def sparse_model(self, sparsity):
        '''
        prune model to speed test
        '''
        alpha = 0.75  # balance number
        # get prune number
        prune_param = (self.per_head_param*self.self_attn.num_heads+self.per_neuron_param*self.mlp.intermediate_size)*sparsity
        prune_attndim_num = int(round(self.self_attn.hidden_size*sparsity*alpha)) # 
        pruned_param = prune_attndim_num*(self.per_head_param/self.self_attn.head_dim) + prune_attndim_num*self.mlp.intermediate_size*3 # 已经移除的
        prune_interdim_num = int((prune_param - pruned_param)/(self.mlp.hidden_size-prune_attndim_num)/3)

        # prune head
        self.compressed_dim   -= prune_attndim_num 
        for w in [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj]:
            weight = w.weight.data
            w.weight.data = weight[:, :-prune_attndim_num]
        w = self.self_attn.o_proj
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num]

        # prune mlp 
        self.mlp.intermediate_size -= prune_interdim_num
        for w in [self.mlp.gate_proj, self.mlp.up_proj]:
            weight = w.weight.data
            w.weight.data = weight[:self.mlp.intermediate_size, :-prune_attndim_num]
        w = self.mlp.down_proj
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num,:self.mlp.intermediate_size]

        # prune_norm 
        w = self.input_layernorm
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num]
        w = self.post_attention_layernorm
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num]


    def forward(self, hidden_states, attention_mask = None, position_ids = None, past_key_value = None, output_attentions = False, use_cache = False, cache_position = None, position_embeddings = None, **kwargs):
        hidden_states = hidden_states[:, :, :self.compressed_dim]
        return super().forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings, **kwargs)