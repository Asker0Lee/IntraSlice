import torch
import time
import os
import shutil
import numpy as np
from .utils import collect_activation_H, collect_mask_grads
from .sobp_utils import search_mask_by_param as sobp_search_mask

def compute_grad_activate(model, dataloder,save_path, is_pca):
    full_head_mask = torch.ones(model.config.num_hidden_layers, model.config.hidden_size, dtype=model.dtype).cuda()
    full_neuron_mask = torch.ones(model.config.num_hidden_layers, model.config.intermediate_size, dtype=model.dtype).cuda()

    activate_head_H = torch.torch.zeros(model.config.num_hidden_layers, model.config.hidden_size, model.config.hidden_size, dtype=torch.float).cuda()
    activate_neuron_H = torch.torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size, model.config.intermediate_size, dtype=torch.float).cuda()

    start = time.time()
    # Search the optimal mask
    collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        dataloder,
        save_path = save_path
    )
    print('get gradient done!')
    if is_pca:
    # 搜集activation Hessian
        collect_activation_H(
            model,
            activate_head_H,
            activate_neuron_H,
            dataloder,
            save_path = save_path
        )
        print('get hessian matrix done!')
    print(time.time() - start)
    

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
    is_attention_mac=False
):
    if is_attention_mac:
        return 4 * seq_len * attention_head_size*hidden_size + attention_head_size*seq_len*seq_len*2
    else:
        return 4 * seq_len * attention_head_size*hidden_size

def mac_per_neuron(seq_len, hidden_size):
    return 3 * seq_len * hidden_size


def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len,
    hidden_size,
    attention_head_size,
    is_attention_mac=False
):
    mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(seq_len, hidden_size, attention_head_size, is_attention_mac)
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size)
        mac += attention_mac + ffn_mac
    return mac





def get_Q_(H, size=128):
    '''
    
    '''
    d = H.shape[-1]
    Q = torch.zeros_like(H)
    V = torch.zeros_like(H[:, 0])
    for l in range(H.shape[0]):
        step = int(d//size)
        for i in range(step):
            X_eig = torch.linalg.eigh(H[l, i*size:i*size+size, i*size:i*size+size].cuda())
            index = torch.argsort(X_eig[0], descending=True)
            eigen_vec = X_eig[1][:, index]
            eigen_val = X_eig[0][index]
            Q[l, i*size:i*size+size, i*size:i*size+size] = eigen_vec
            V[l, i*size:i*size+size] = eigen_val**0.5/torch.sum(eigen_val**0.5)
    H=H.cpu()
    return Q.float().cpu(), V.float().cpu()

def get_Qtrans_neuron_grad(grads, H, size=128):
    '''
    grads (B, layers, dim)
    H (layers, dim, dim)
    '''
    #(layers,B, dim)
    grads = grads.float().transpose(0, 1) 
    for l in range(H.shape[0]): 
        # get layer Q
        score = torch.diag(H[l])
        _, sorted_indices = torch.topk(score.view(-1), k=H.shape[1])
        keep_num = H.shape[1] - size
        Q = torch.zeros(size=(H.shape[1], H.shape[1]), dtype=H.dtype).cuda()
        Q[sorted_indices[:keep_num], range(keep_num)] = 1  
        grads[l]=(grads[l].cuda()@Q.cuda()).cpu()
        # grads[l]=(grads[l].cuda()@Q[l].cuda()).cpu()
    return torch.transpose(grads, 0, 1)


def get_Qtrans_grad(grads, H, size=128):
    '''
    grads (B, layers, dim)
    H (layers, dim, dim)
    '''
    Q, V = get_Q_(H, size)  # (layers, dim, dim)
    grads = grads.float().transpose(0, 1) #(layers,B, dim)
    for l in range(H.shape[0]): 
        grads[l]=(grads[l].cuda()@Q[l].cuda()).cpu()
        # grads[l]=(grads[l].cuda()@Q[l].cuda()).cpu()
    return torch.transpose(grads, 0, 1)

def check_files(path, is_pca):
    file_list = os.listdir(path)
   
    if 'head_grads.pt' not in file_list:
        return False
    if 'neuron_grads.pt' not in file_list:
        return False
    if is_pca:
        if 'activate_head_H.pt' not in file_list:
            return False
        if 'activate_neuron_H.pt' not in file_list:
            return False

    return True



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


@torch.no_grad()
def search_mac_pca(
    head_importance,
    neuron_importance,
    seq_len=1,
    mac_constraint=0.7,
    bias_rate = 0.99,  # 
    config=None,
    frac=4,      
    is_attention_mac=False


):
    '''
 
    '''
    assert mac_constraint < 1
   
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads*frac
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads) 
        
    # original_mac = hidden_size*hidden_size*4*32 + hidden_size*intermediate_size*3*32
    original_mac = compute_mac(num_heads_per_layer=[num_attention_heads] * num_hidden_layers,
                                num_neurons_per_layer=[intermediate_size] * num_hidden_layers,
                                seq_len=seq_len,
                                hidden_size=hidden_size,
                                attention_head_size=attention_head_size,is_attention_mac=is_attention_mac)
    per_head_mac = mac_per_head(seq_len, hidden_size, attention_head_size) # attention中每个剪枝单元的计算量
    per_neuron_mac = mac_per_neuron(seq_len, hidden_size)#ffn中每个剪枝单元的计算量
    max_mac = mac_constraint * original_mac

    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)

    #
    num_heads = int(num_attention_heads*num_hidden_layers*mac_constraint*bias_rate)  # 选择多少attention block的单元
    heads_mac = per_head_mac*num_heads
    neurons_mac = max_mac - heads_mac
    num_neurons = int(neurons_mac /per_neuron_mac)
    num_neurons = max(num_neurons, 0)
    head_indicies = sorted_head_indicies[:num_heads]
    neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_mask = torch.zeros(num_hidden_layers * num_attention_heads).cpu()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).cpu()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    return head_mask, neuron_mask



@torch.no_grad()
def search_param_pca(
    head_importance,
    neuron_importance,
    mac_constraint=0.7,
    bias_rate = 0.99,  
    config=None,
    frac=4,     
):
    '''
    在grad经过pca处理后，通用global 剪枝率分析方法
    '''
    assert mac_constraint < 1
   
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads*frac
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    
        
    # original_mac = hidden_size*hidden_size*4*32 + hidden_size*intermediate_size*3*32
    
    per_head_param = param_per_head(config)/frac 
    per_neuron_param = param_per_neuron(config)
    original_param = num_hidden_layers*(per_head_param*num_attention_heads + per_neuron_param*intermediate_size)
    max_param = mac_constraint * original_param

    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)
    
    #select 
    num_neurons = int(num_hidden_layers*intermediate_size*(1-(1-mac_constraint)*bias_rate))  
    neurons_param = num_neurons*per_neuron_param
    head_param = max_param - neurons_param
    num_heads = int(head_param/per_head_param)
    num_heads = max(num_heads, 0)

    head_indicies = sorted_head_indicies[:num_heads]
    neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_mask = torch.zeros(num_hidden_layers * num_attention_heads).cpu()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).cpu()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    return head_mask, neuron_mask





import matplotlib.pyplot as plt
import numpy as np
def show_layers_prune(prune_data, save_path):
    plt.figure(figsize=(10,5))
    prune_data = list(np.reshape(prune_data, (-1)))
    x = range(len(prune_data))
    plt.plot(x, prune_data, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

  
    for i in range(len(prune_data)):
        if i % 2 == 1:  
            plt.scatter(i, prune_data[i], color='red', s=50, label='Odd Index' if i == 1 else "")
        else:  # 
            plt.scatter(i, prune_data[i], color='blue', s=50, label='Even Index' if i == 0 else "")
    plt.axis([0,len(prune_data), 0, 1])
    plt.savefig(save_path)



def recorrect_prune_rate(prune_rate, max_prune_rate=0.9):
    '''
        prune_rate (torch.Tensor): 形状为(N, 1)的剪枝率张量
        max_prune_rate (float): 最大允许的剪枝率，默认为0.9
        
    '''
    if not isinstance(prune_rate, torch.Tensor):
        prune_rate = torch.tensor(prune_rate, dtype=torch.float32)
    
    
    over_mask = prune_rate > max_prune_rate
    total_reduce = (prune_rate[over_mask] - max_prune_rate).sum()
    prune_rate[over_mask] = max_prune_rate
    
    if total_reduce <= 1e-6: 
        return prune_rate
    
  
    under_mask = (prune_rate < max_prune_rate) & (prune_rate > 0)
    under_values = prune_rate[under_mask]
    
    if not under_mask.any():  
        return prune_rate
    
   
    sorted_values, sorted_indices = torch.sort(under_values, descending=True)
    remaining_to_assign = total_reduce
    
   
    for i in range(len(sorted_values)):
        current_max = sorted_values[i]
        remaining_values = sorted_values[i:]
        sum_remaining = remaining_values.sum()
        
       
        max_possible_add = max_prune_rate - current_max
        
        if sum_remaining > 0:
           
            proportion = current_max / sum_remaining
            assign_amount = min(remaining_to_assign * proportion, max_possible_add)
        else:
            assign_amount = 0
        
     
        sorted_values[i] += assign_amount
        remaining_to_assign -= assign_amount
        
        if remaining_to_assign <= 1e-6:
            break
   
    prune_rate[under_mask] = sorted_values.gather(0, sorted_indices.argsort())
    
    return prune_rate


## pca-based global prune rate 
def get_global_prune_rate(model, dataloader, args, logger=None):
    is_pca = args.global_layer_rate == 'global_pca'
    model_name = args.model.split('/')[-1]
    calib_data = args.dataset
    logger.info(f'############## Get Global Non-uinform Pruning Ratio #####################')
    grad_activate_path = f'grad_activate_results/{model_name}_{calib_data}_result'
    figure_save_path = os.path.join(f'grad_activate_results/global_prune_figure', f"{model_name}_{calib_data}")
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    if not os.path.exists(grad_activate_path):# 
        
        compute_grad_activate(model=model,dataloder=dataloader, save_path=grad_activate_path, is_pca=is_pca)
    elif  not check_files(grad_activate_path, is_pca): #
     
        shutil.rmtree(grad_activate_path)
        compute_grad_activate(model=model,dataloder=dataloader, save_path=grad_activate_path, is_pca=is_pca)
       

    
    
    # load grad and activate
   
    head_grads = torch.load(os.path.join(grad_activate_path, 'head_grads.pt')).float() # shape is (B, layers, dim)
    neuron_grads = torch.load(os.path.join(grad_activate_path, 'neuron_grads.pt')).float() # shape is (B, layers, inter_dim)
    if is_pca:
        head_H = torch.load(os.path.join(grad_activate_path, 'activate_head_H.pt')) # shape is (layers, dim, dim)
        neuron_H = torch.load(os.path.join(grad_activate_path, 'activate_neuron_H.pt')) # shape is (layers, inter_dim, inter_dim)
    else:
         logger.info(f'--unuse pca')
    logger.info(f'--load done!')
   
    config = model.config
    B, L, D = head_grads.shape
    B2,L2,D2 = neuron_grads.shape
    group =8 # Merging gradients to improve stability
    if group !=1:
        head_grads = torch.sum(torch.reshape(head_grads, shape=(-1, group, L, D)), dim=1)
        neuron_grads = torch.sum(torch.reshape(neuron_grads, shape=(-1, group, L2, D2)), dim=1)
        B = int(B/group)
    frac = args.global_frac
    bias_rate =args.global_bias
    head_dim = int(config.hidden_size/config.num_attention_heads)
    block_size = [128, int(config.intermediate_size/32)]
    if is_pca:
        head_grads2 = get_Qtrans_grad(head_grads, head_H, size=block_size[0])
        # neuron_grads2 = get_Qtrans_grad(neuron_grads, neuron_H, size=block_size[1])
        neuron_grads2 = neuron_grads
        head_grads_g = torch.reshape(head_grads2, shape=(B, L, -1, int(head_dim/frac))).sum(-1)  # 
        neuron_grads2 = neuron_grads2.pow(2).sum(dim=0)
        head_grads_g = head_grads_g.pow(2).sum(dim=0)
    else:
        head_grads2 = head_grads
        neuron_grads2 = neuron_grads

        head_grads_g = torch.reshape(head_grads2, shape=(B, L, -1, int(head_dim/frac))).sum(-1)  # 
        neuron_grads2 = neuron_grads2.pow(2).sum(dim=0)
        head_grads_g = head_grads_g.pow(2).sum(dim=0)
   
    # head_grads2 = head_grads2.pow(2).sum(dim=0)
    
    head_mask, neuron_mask = search_param_pca(head_grads_g, neuron_grads2,  mac_constraint=1-args.sparsity, frac=frac, bias_rate=bias_rate, config=config)

    # head_mask, neuron_mask = sobp_search_mask(model=model, importance={'head':head_grads_g, 'neuron':neuron_grads2},  mac_constraint=1-args.sparsity)
    logger.info(f'--get global pruning ratio done!')
    pt= []
    for i in range(config.num_hidden_layers):
        attn_prune = (1-torch.sum(head_mask[i])/(config.num_attention_heads*frac)).item()
        mlp_prune = (1-torch.sum(neuron_mask[i])/config.intermediate_size).item()
        pt.append([attn_prune, mlp_prune]) 
    pt = torch.Tensor(pt).float()

   
    pt[:, 0] = recorrect_prune_rate(pt[:, 0], max_prune_rate=0.8)
    pt[:, 1] = recorrect_prune_rate(pt[:, 1], max_prune_rate=0.8)
    
    file_name = os.path.join(figure_save_path, f'sparsity={args.sparsity}_blcok_size={block_size}_pca={is_pca}_frac={frac}_bias={bias_rate}_group={group}_sobp_.png')
    show_layers_prune(pt.numpy(), file_name)
    logger.info(f'--the picture of prune ratio saved in {file_name}')
    return pt