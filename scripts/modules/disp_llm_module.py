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

class disp_LlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx = None):
        
        super().__init__(config, layer_idx)

        
     
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






class DISP_LlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, config, layer_idx, sparsity):
        config._attn_implementation = 'eager'
        super().__init__(config, layer_idx)
        self.self_attn = disp_LlamaAttention(config, layer_idx)
        self.register_buffer('select_1', None)
        self.register_buffer('select_2', None)
        self.register_buffer('select_3', None)
        self.register_buffer('select_4', None)
        self.register_buffer('select_5', None)
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
        alpha = [1.8,  1.0, 0.2, 0.5]  # balance number
        s1 = sparsity*alpha[0]
        s2 = sparsity*alpha[1]
        s3 = sparsity*alpha[2]
        s5 = sparsity*alpha[3]
        # get prune number
        hidden_size = self.self_attn.hidden_size
        attention_head_size = self.self_attn.head_dim
        prune_param = (self.per_head_param*self.self_attn.num_heads+self.per_neuron_param*self.mlp.intermediate_size)*sparsity # all prune param number
        prune_attndim_num_s1 = int(round(self.self_attn.hidden_size*s1)) # 
        prune_attndim_num_s2 = int(round(self.self_attn.hidden_size*s2))
        mlp_input_purne_num_s3 = int(round(self.self_attn.hidden_size*s3))
        mlp_out_prune_num_s5 = int(round(self.self_attn.hidden_size*s5))
        per_qkv_dim_param = hidden_size + 2/self.self_attn.num_key_value_groups*hidden_size
        pruned_param = prune_attndim_num_s1*per_qkv_dim_param + prune_attndim_num_s2*hidden_size + mlp_input_purne_num_s3*2*self.mlp.intermediate_size + mlp_out_prune_num_s5*self.mlp.intermediate_size

        # pruned_param = prune_attndim_num*(self.per_head_param/self.self_attn.head_dim) + prune_attndim_num*self.mlp.intermediate_size*3 # 已经移除的
        mlp_inter_prune_num_s4 = int((prune_param - pruned_param)/((hidden_size - mlp_input_purne_num_s3)*2 + (hidden_size - mlp_out_prune_num_s5)))

        # prune head

        self.select_1 = torch.randperm(self.self_attn.hidden_size)[:(self.self_attn.hidden_size-prune_attndim_num_s1)]
        self.select_2 = torch.randperm(self.self_attn.hidden_size)[:(self.self_attn.hidden_size-prune_attndim_num_s2)]
        self.select_3 = torch.randperm(self.self_attn.hidden_size)[:(self.self_attn.hidden_size-mlp_input_purne_num_s3)]
        self.select_4 = torch.randperm(self.mlp.intermediate_size)[:(self.mlp.intermediate_size-mlp_inter_prune_num_s4)]
        self.select_5 = torch.randperm(self.self_attn.hidden_size)[:(self.self_attn.hidden_size-mlp_out_prune_num_s5)]

        for w in [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj]:
            weight = w.weight.data
            w.weight.data = weight[:, :-prune_attndim_num_s1]
        w = self.self_attn.o_proj
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num_s2]

        # prune mlp 
        self.mlp.intermediate_size -= mlp_inter_prune_num_s4
        for w in [self.mlp.gate_proj, self.mlp.up_proj]:
            weight = w.weight.data
            w.weight.data = weight[:self.mlp.intermediate_size, :-mlp_input_purne_num_s3]
        w = self.mlp.down_proj
        weight = w.weight.data
        w.weight.data = weight[:-mlp_out_prune_num_s5,:self.mlp.intermediate_size]

        # prune_norm 
        w = self.input_layernorm
        weight = w.weight.data
        w.weight.data = weight[:-prune_attndim_num_s1]
        w = self.post_attention_layernorm
        weight = w.weight.data
        w.weight.data = weight[:-mlp_input_purne_num_s3]


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        if self.select_1 is not None:
            hidden_states_ = hidden_states[:,:,self.select_1]
        else:
            hidden_states_ = hidden_states

        hidden_states_ = self.input_layernorm(hidden_states_)

        # Self Attention
        hidden_states_, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states_,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # hidden_states = residual + hidden_states
        if self.select_2 is not None:
            hidden_states.scatter_add_(dim=-1, index=self.select_2.unsqueeze(0).unsqueeze(0).expand(bsz,q_len, -1), src=hidden_states_)
        
        # Fully Connected
        residual = hidden_states
        if self.select_3 is not None:
            hidden_states_ = hidden_states[:,:,self.select_3]
        else:
            hidden_states_ = hidden_states
        hidden_states_ = self.post_attention_layernorm(hidden_states_)
        hidden_states_ = self.mlp(hidden_states_)
        # hidden_states = residual + hidden_states
        if self.select_5 is not None:
            hidden_states.scatter_add_(dim=-1, index=self.select_5.unsqueeze(0).unsqueeze(0).expand(bsz,q_len, -1), src=hidden_states_)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs