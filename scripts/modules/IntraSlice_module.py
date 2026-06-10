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


def repeat_kv_mask(hidden_states: torch.Tensor, n_rep: Union[int,list,torch.Tensor]) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if isinstance(n_rep,int):
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    else:
        # expanded = hidden_states.unsqueeze(2)
        # expanded_list = []
        # num_key_value_heads = len(n_rep)
        # for i in range(num_key_value_heads):
        #     rep = n_rep[i]
        #     expanded_head = expanded[:, i:i+1, :, :, :]
        #     expanded_head = expanded_head.expand(batch, 1, rep, slen, head_dim)
        #     expanded_list.append(expanded_head)
        # hidden_states = torch.cat(expanded_list, dim=2).reshape(batch, -1, slen, head_dim)
        hidden_states = expand_interval(hidden_states=hidden_states, n_rep=n_rep)
        return  hidden_states

def expand_interval(hidden_states: torch.Tensor, n_rep: list[int]):
    """
    hidden_states: [B, H_kv, L, D]
    n_rep: list[int] len is H_kv, denotes repeate-number of each head
    return: expanded_hidden_states: [B, sum(n_rep), L, D]
    """
    B, H_kv, L, D = hidden_states.shape
    expanded = hidden_states.unsqueeze(2)  # [B, H_kv, 1, L, D]
    expanded_list = []

    for i, rep in enumerate(n_rep):
        head_i = expanded[:, i:i+1, :, :, :].expand(B, 1, rep, L, D)
        expanded_list.append(head_i)
    out = torch.cat(expanded_list, dim=2).reshape(B, sum(n_rep), L, D)
    return out



def apply_rotary_pos_emb_masks_fast(q, k, cos, sin, masks, group, unsqueeze_dim=1):
    """
    q, k: [B, H, S, D'] — partial dims
    cos, sin: [1, S, D_rope] — full RoPE
    masks: [H, 1, D_rope] — mask selecting which RoPE dims are used in current q/k
    """
    if isinstance(group,int):
        B, H, S, D_used = q.shape
        _, _, D_rope = cos.shape
        selected_indices = torch.topk(masks, k=D_used, dim=-1).indices     # [H, 1, D_used]
        selected_indices = selected_indices.sort(dim=-1).values            # 保证顺序一致
        index = selected_indices.expand(H, S, -1)                          # [H, S, D_used]
        cos_sel = cos[0].expand(H, -1, -1).gather(2, index)                # [H, S, D_used]
        sin_sel = sin[0].expand(H, -1, -1).gather(2, index)                # [H, S, D_used]
        q_rot = q * cos_sel.unsqueeze(0) + rotate_half(q) * sin_sel.unsqueeze(0)
        k_rot = k * cos_sel.unsqueeze(0) + rotate_half(k) * sin_sel.unsqueeze(0)
    else:
        B, H, S, D_used = k.shape
        _, _, D_rope = cos.shape
        selected_indices = torch.topk(masks, k=D_used, dim=-1).indices     # [H, 1, D_used]
        selected_indices = selected_indices.sort(dim=-1).values            # 保证顺序一致
        index = selected_indices.expand(H, S, -1)                          # [H, S, D_used]
        cos_sel = cos[0].expand(H, -1, -1).gather(2, index)                # [H, S, D_used]
        sin_sel = sin[0].expand(H, -1, -1).gather(2, index)                # [H, S, D_used]
        # expand for q
        cos_sel_q = expand_interval(cos_sel.unsqueeze(0), group)[0]
        sin_sel_q = expand_interval(sin_sel.unsqueeze(0), group)[0]
        q_rot = q * cos_sel_q.unsqueeze(0) + rotate_half(q) * sin_sel_q.unsqueeze(0)
        k_rot = k * cos_sel.unsqueeze(0) + rotate_half(k) * sin_sel.unsqueeze(0)

    return q_rot, k_rot


class intraslice_LlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx = None):
        super().__init__(config, layer_idx)
        self.register_buffer('rope_mask', None)

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
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.rope_mask is not None:
            query_states, key_states = apply_rotary_pos_emb_masks_fast(query_states, key_states, cos, sin, masks=self.rope_mask, group=self.num_key_value_groups)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = torch.cat([past_key_value.key_cache[self.layer_idx], key_states], dim=-2), torch.cat([past_key_value.value_cache[self.layer_idx], value_states], dim=-2)

        key_states = repeat_kv_mask(key_states, self.num_key_value_groups)
        value_states = repeat_kv_mask(value_states, self.num_key_value_groups)

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

class IntraSliceLlamaDecoderLayer(LlamaDecoderLayer):
    """
    This class simulates the LlamaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L376)
    but with the addition of a shortcut_Q attribute. This attribute is used to rotate the residual tensors.
    """
    def __init__(self, config, layer_idx, sparsity):
        config._attn_implementation = 'eager'
        super().__init__(config, layer_idx)
        self.self_attn = intraslice_LlamaAttention(config, layer_idx)
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
        # get prune number
        alpha = 1.0
        prune_param = (self.per_head_param*self.self_attn.num_heads+self.per_neuron_param*self.mlp.intermediate_size)*sparsity
        prune_head_num = int(round(self.self_attn.num_heads*sparsity*alpha*0.8))
        compressed_dim =self.self_attn.head_dim - int((round(self.self_attn.hidden_size*sparsity*alpha) - prune_head_num*self.self_attn.head_dim)//(self.self_attn.num_heads-prune_head_num)//8*8)
        prune_attndim_num =self.self_attn.hidden_size - compressed_dim*(self.self_attn.num_heads-prune_head_num)

        prune_interdim_num = int((prune_param - prune_attndim_num*self.self_attn.hidden_size*4)//self.per_neuron_param)

        # prune head
        if self.self_attn.num_heads == self.self_attn.num_key_value_heads:
            self.self_attn.rope_mask = torch.zeros(size=(self.self_attn.num_heads-prune_head_num, 1, self.self_attn.head_dim))
            self.self_attn.num_heads -= prune_head_num
            self.self_attn.num_key_value_heads -= prune_head_num
            self.self_attn.head_dim = compressed_dim
            self.self_attn.hidden_size = compressed_dim*self.self_attn.num_heads
            indice = torch.randperm(self.self_attn.head_dim//2)[:compressed_dim//2]
            self.self_attn.rope_mask[:,:,indice]=1
            self.self_attn.rope_mask[:,:,indice+compressed_dim//2]=1
            for w in [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj]:
                weight = w.weight.data
                w.weight.data = weight[:-prune_attndim_num]
            
        else: # GQA
            prune_head_index = torch.randperm(self.self_attn.num_heads)[:prune_head_num]
            keep_head_group = torch.ones(size=(self.self_attn.num_heads, ))
            keep_head_group[prune_head_index] = 0 # prune heads are 0
            keep_head_group = torch.reshape(keep_head_group, shape=(-1, self.self_attn.num_key_value_groups)) 
            keep_head_group_s = keep_head_group.sum(-1)
            key_value_pruned_head_num = torch.sum(keep_head_group_s == 0)
            
            self.self_attn.num_key_value_groups= [int(keep_head_group_s[i]) for i in range(self.self_attn.num_key_value_heads) if keep_head_group_s[i] > 0]
            self.self_attn.num_heads -= prune_head_num
            self.self_attn.num_key_value_heads -= key_value_pruned_head_num
            self.self_attn.rope_mask = torch.zeros(size=(self.self_attn.num_key_value_heads, 1, self.self_attn.head_dim)) # when used in query, rope_mask need to expand
            self.self_attn.head_dim = compressed_dim
            self.self_attn.hidden_size = compressed_dim*self.self_attn.num_heads
            indice = torch.randperm(self.self_attn.head_dim//2)[:compressed_dim//2]
            self.self_attn.rope_mask[:,:,indice]=1
            self.self_attn.rope_mask[:,:,indice+compressed_dim//2]=1

            for w in [self.self_attn.q_proj]:
                weight = w.weight.data
                w.weight.data = weight[:-prune_attndim_num]
            for w in [self.self_attn.k_proj, self.self_attn.v_proj]:
                weight = w.weight.data
                w.weight.data = weight[:self.self_attn.num_key_value_heads*self.self_attn.head_dim]

        w = self.self_attn.o_proj
        weight = w.weight.data
        w.weight.data = weight[:, :-prune_attndim_num]

        # prune mlp 
        self.mlp.intermediate_size -= prune_interdim_num
        for w in [self.mlp.gate_proj, self.mlp.up_proj]:
            weight = w.weight.data
            w.weight.data = weight[:-prune_interdim_num]
        w = self.mlp.down_proj
        weight = w.weight.data
        w.weight.data = weight[:, :-prune_interdim_num]