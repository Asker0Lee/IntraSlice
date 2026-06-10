# IntraSlice

IntraSlice is a framework for structured pruning of large language models (LLMs) via block-wise module-intra PCA compression. By leveraging the structural characteristics of Transformer modules, IntraSlice introduces an approximate PCA method whose transformation matrices can be fully fused into the model without introducing any additional parameters.
Additionally, a global pruning ratio estimator based on PCA is proposed to better capture the distributional changes in compressed activations, improving upon traditional module importance metrics.


## Project Structure
```
.
├── README.md                  # Project description
├── evaluation/                # Evaluation code (zero-shot tasks)
├── pruning_src/                    
│   ├── adapters/              # Pruning adpter of LLaMA and Phi model 
|   ├── global_prune_rate/          # Globa non-uninform prunging ratio evaluation code
|   ├── disp_eval_utils.py     # Dataset loader and ppl test code of DISP-LLM method
│   ├── data_utils.py          # Dataset loader and ppl test code
│   ├── prune_method.py          # pruning code
│   ├── prune_utils_llama.py          # pruning class for llama
│   └── prune_utils_phi3.py          # pruning class for phi3
├── scripts/ 
│   ├── modules/              # DecodeLayer for pruned modules
│   └── speedup.py          # speedup test for pruned modules

├── run_IntraSlice.py          # Main script for pruning
├── run-purne.sh               # Main script for pruning

```


## How to Prune
Run the following command or run `run-purne.sh` directly.
```
CUDA_VISIBLE_DEVICES=0 apython run_InterSlice.py --model meta-llama/Llama-2-7b-hf   --sparsity 0.30  --dataset wikitext2 --global_bias 1.0
```

## Speedup Test
To accurately measure the speedup of different methods, we tested a single decoder layer to remove the effects of embedding and random sampling. Sparsity is the average of all layers. For all speed experiments, the prefill speed is evaluated with a 4096-length inputs. The generation speed is evaluated with a 4096-length KV cache. 

You can test the speed by run `scripts/speedup.py` directly.
