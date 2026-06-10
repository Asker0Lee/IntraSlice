import torch 
from pruning_src.utils import get_model_properties
from collections import defaultdict
import math
def param_per_head(
    config
):
    architectures = config.architectures
    hidden_size = config.hidden_size
    attention_head_size = config.hidden_size//config.num_attention_heads

    if 'LlamaForCausalLM' in architectures:
        num_key_value_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        per_head_qkv = hidden_size*attention_head_size + 2*num_key_value_heads/num_heads*hidden_size*attention_head_size
        per_head_output = hidden_size*attention_head_size
        param = per_head_qkv+per_head_output
    elif 'PhiForCausalLM' in architectures :
        num_key_value_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        per_head_qkv = hidden_size*attention_head_size + 2*num_key_value_heads/num_heads*hidden_size*attention_head_size + attention_head_size + 2*num_key_value_heads/num_heads*hidden_size
        per_head_output = hidden_size*attention_head_size + attention_head_size
        param = per_head_qkv+per_head_output
    elif 'Phi3ForCausalLM' in architectures:
        num_key_value_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        per_head_qkv = hidden_size*attention_head_size + 2*num_key_value_heads/num_heads*hidden_size*attention_head_size
        per_head_output = hidden_size*attention_head_size
        param = per_head_qkv+per_head_output
    return param



def param_per_neuron(config):
    architectures = config.architectures
    hidden_size = config.hidden_size
    if 'LlamaForCausalLM' in architectures:
        return 3*hidden_size
    elif 'Phi3ForCausalLM' in architectures:
        return 3*hidden_size  #up\gate\down
    elif 'PhiForCausalLM' in architectures:
        return 2*hidden_size




def constrained_indicies(model,sorted_head_indicies,num_heads,\
                             sorted_neuron_indicies,num_neurons,head_remain_ratio=0.1,neuron_remain_ratio=0.02):
    device = sorted_head_indicies.data.device

    num_hidden_layers, num_attention_heads, ffn_dim, hidden_size, attention_head_size = \
        get_model_properties(model)

    head_indicies = sorted_head_indicies[:num_heads]
    neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_idle_seat = {}
    new_head_indicies = {}
    neuron_idle_seat = {}
    new_neuron_indicies = {}
    # the max para can be  pruned
    max_pruned_heads = math.ceil(num_attention_heads * (1-head_remain_ratio))
    max_pruned_neurons = math.ceil(ffn_dim * (1-neuron_remain_ratio))

    excess_heads = 0
    excess_neurons = 0
    for i in range(num_hidden_layers):
        # head
        head_low_bound = i * num_attention_heads - 1
        head_up_bound = (i + 1) * num_attention_heads
        bool_indicies = torch.gt(head_indicies,head_low_bound) & \
                   torch.lt(head_indicies,head_up_bound)
        pruned_heads = torch.sum(bool_indicies)
        indicies = torch.nonzero(bool_indicies).squeeze(dim=1)
        assert pruned_heads.item() == indicies.shape[0], "something wrong in constrained_indicies"
        if pruned_heads > max_pruned_heads:
            head_idle_seat[i] = 0
            excess_num = pruned_heads - max_pruned_heads
            excess_heads += excess_num
            indicies = indicies[:max_pruned_heads]
        else:
            head_idle_seat[i] = max_pruned_heads - pruned_heads
        new_head_indicies[i] = head_indicies[indicies]

        # neuron
        neuron_low_bound = i * ffn_dim - 1
        neuron_up_bound = (i + 1) * ffn_dim
        bool_indicies = torch.gt(neuron_indicies, neuron_low_bound) & \
                   torch.lt(neuron_indicies,neuron_up_bound)
        pruned_neurons = torch.sum(bool_indicies)
        # # 保证是8的倍数有利于推理加速
        # pruned_neurons = pruned_neurons - pruned_neurons%8

        indicies = torch.nonzero(bool_indicies).squeeze(dim=1)
        assert pruned_neurons.item() == indicies.shape[0], "something wrong in constrained_indicies"
        if pruned_neurons  > max_pruned_neurons:
            neuron_idle_seat[i] = 0
            excess_num = pruned_neurons - max_pruned_neurons
            excess_neurons += excess_num
            indicies = indicies[:max_pruned_neurons]
        else:
            neuron_idle_seat[i] = max_pruned_neurons - pruned_neurons
        new_neuron_indicies[i] = neuron_indicies[indicies]


    new_head_candidate = defaultdict(list)
    new_neuron_candidate = defaultdict(list)

    while excess_heads>0:
        prune_idx = sorted_head_indicies[num_heads]
        idx = int(prune_idx/num_attention_heads)
        if head_idle_seat[idx]>0:
            new_head_candidate[idx].append(prune_idx)
            head_idle_seat[idx] -= 1
            excess_heads -= 1
        num_heads += 1

    while excess_neurons>0:
        prune_idx = sorted_neuron_indicies[num_neurons]
        idx = int(prune_idx/ffn_dim)
        if neuron_idle_seat[idx]>0:
            new_neuron_candidate[idx].append(prune_idx)
            neuron_idle_seat[idx] -= 1
            excess_neurons -= 1
        num_neurons += 1

    for i in range(num_hidden_layers):
        if i in new_head_candidate:
            new_head_indicies[i] = torch.cat((new_head_indicies[i],torch.tensor(new_head_candidate[i],device=device)),dim=0)
        if i in new_neuron_candidate:
            new_neuron_indicies[i] = torch.cat((new_neuron_indicies[i],torch.tensor(new_neuron_candidate[i],device=device)),dim=0)


    head_indicies = torch.cat([new_head_indicies[key] for key in new_head_indicies],dim=0)
    neuron_indicies = torch.cat([new_neuron_indicies[key] for key in new_neuron_indicies],dim=0)


    return head_indicies,neuron_indicies

def search_mask_by_param(model,importance=None, mac_constraint=None,layers_mask=None):

    num_layers, num_attention_heads, ffn_dim, hidden_size, head_size = \
        get_model_properties(model)
    config = model.config
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads*1
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size


    per_head_param = param_per_head(config)/1 # attention中每个剪枝单元的参数量，frac是将head分成几份来考虑，细粒度
    per_neuron_param = param_per_neuron(config)#ffn中每个剪枝单元的参数量

    original_param = num_hidden_layers*(per_head_param*num_attention_heads + per_neuron_param*intermediate_size)
    delete_para = (1-mac_constraint) * original_param
    layers_mask = {}
    layers_mask['neuron_mask'] = torch.ones((num_hidden_layers, ffn_dim), dtype=torch.float16,device='cuda')
    layers_mask['head_mask'] = torch.ones((num_hidden_layers, num_attention_heads),dtype=torch.float16, device='cuda')
    head_masks = layers_mask['head_mask']
    neuron_masks = layers_mask['neuron_mask']
    valid_head_masks = abs(head_masks)>1e-7
    valid_neuron_masks = abs(neuron_masks)>1e-7

    head_importance = importance['head'].cuda()
    neuron_importance = importance['neuron'].cuda() * 250*per_neuron_param / per_head_param

    unpruned_head_importance = head_importance[valid_head_masks]
    unpruned_neuron_importance = neuron_importance[valid_neuron_masks]

    sorted_head_importance, sorted_head_indicies = unpruned_head_importance.view(-1).sort(descending=False)
    sorted_neuron_importance, sorted_neuron_indicies = unpruned_neuron_importance.view(-1).sort(descending=False)

    min_importance = float('inf')
    head_remain_ratio = 0.2
    neuron_remain_ratio = 0.2
    max_prune_heads = int(torch.sum(valid_head_masks).item() * (1 - head_remain_ratio))
    max_prune_neurons = int(torch.sum(valid_neuron_masks).item() * (1 - neuron_remain_ratio))

    per_head_para = param_per_head(model.config)
    per_neuron_para = param_per_neuron(model.config)

    for num_heads in range(max_prune_heads + 1):
        heads_param = per_head_para * num_heads
        neurons_param = delete_para - heads_param
        if neurons_param<0:
            num_heads = int(delete_para/per_head_para)
        num_neurons = int(neurons_param / per_neuron_para)
        num_neurons = max(num_neurons, 0)
        num_neurons = min(max_prune_neurons, num_neurons)

        # 控制每层的剪枝率,避免整层崩溃
        h_indicies, n_indicies = constrained_indicies(model, sorted_head_indicies, num_heads, \
                                                      sorted_neuron_indicies, num_neurons,
                                                      head_remain_ratio,neuron_remain_ratio)
        total_importance = unpruned_head_importance[h_indicies].sum() + unpruned_neuron_importance[n_indicies].sum()
        if total_importance < min_importance:
            min_importance = total_importance
            head_indicies = h_indicies
            neuron_indicies = n_indicies

        if neurons_param < 0:
            break

    new_head_mask = torch.ones(num_layers * num_attention_heads,dtype=torch.float16).cuda()
    new_head_mask[~valid_head_masks.view(-1)] = 0.0
    unpruned_head = new_head_mask[valid_head_masks.view(-1)]
    unpruned_head[head_indicies] = 0.0
    new_head_mask[valid_head_masks.view(-1)] = unpruned_head
    new_head_mask = new_head_mask.view(num_layers, num_attention_heads)

    new_neuron_mask = torch.ones(num_layers * ffn_dim,dtype=torch.float16).cuda()
    new_neuron_mask[~valid_neuron_masks.view(-1)] = 0.0
    unpruned_neuron = new_neuron_mask[valid_neuron_masks.view(-1)]
    unpruned_neuron[neuron_indicies] = 0.0
    new_neuron_mask[valid_neuron_masks.view(-1)] = unpruned_neuron
    new_neuron_mask = new_neuron_mask.view(num_layers, ffn_dim)

    return new_head_mask, new_neuron_mask

