import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='7'
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
import pathlib
import shutil
import sys
import torch
import time
from pruning_src import utils, hf_utils
from pruning_src import data_utils
from evaluation import eval_uitls
from pruning_src.global_prune_rate.get_global_prune_rate import get_global_prune_rate, show_layers_prune




def slicing_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to load")
    parser.add_argument("--sliced_model_path",type=str,help="Path to sliced model",default=None,)
    # prune config
    parser.add_argument("--dataset",type=str,choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2")
    parser.add_argument("--nsamples",type=int,help="Number of samples of the calibration data to load.",default=128)
    # parser.add_argument("--batch_size", type=int, default=8, help="Batch size for loading the calibration data.")
    # parser.add_argument("--seqlen", type=int, default=2048, help="Maximum sequence length for the calibration data.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument(
        "--sparsity", type=float, default=0.4, help="A measure of how much slicing is applied (in the range [0, 1))")
    parser.add_argument("--distribute_model",action="store_true",    )
    parser.add_argument("--save_dir", type=str, default='save_model', help="Path to save the model.")
    parser.add_argument('--device',type=str,choices=['cpu', 'cuda'],default='cuda',help="")
    parser.add_argument('--iterpca', action="store_true")
    # global prune rate config
    parser.add_argument("--global_layer_rate",type=str,
                        default="global",choices=["global_pca", "global", "uniform"])
    parser.add_argument("--global_bias",type=float,
                        default=1.0)
    parser.add_argument("--global_frac",type=int,
                        default=1, help='The number of blocks to divide a head into')
    
    # evaluation config
    parser.add_argument('--ppl_tasks',nargs='+',default=["wikitext2", "c4"])
    parser.add_argument('--tasks',nargs='+',default=["arc_challenge","arc_easy", "boolq", "hellaswag", "openbookqa", "piqa",   "winogrande"],)
    parser.add_argument("--ppl_eval_seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity.")
    parser.add_argument("--ppl_eval_batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument("--ppl_eval_nsamples", type=int, default=128, help="Number of samples to evaluate the perplexity on.")
    parser.add_argument("--eval_baseline", action="store_false", help="Evaluate the baseline model.")
    parser.add_argument('--eval_zero_shot_task', action="store_false", help="is test zero shot task? default is False",)
    return parser.parse_args() 




def slicing_main(args: argparse.Namespace, logger) -> None:
    logger.info("Running InterSlice experiment.")
    logger.info(f"PyTorch device: {args.device}")
    logger.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    logger.info(f'---------------------------------\n all args is {args} \n ----------------------------------------\n')
    for arg, argv in vars(args).items():
        logger.info(f'{arg} = {argv}')
   

    utils.seed_all(args.seed)

    if args.sliced_model_path:
        # load the model from sliced_model_path to compute perplexity and skip rotation and slicing
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            round_interval=args.round_interval,
            token=args.hf_token,
        )
    else:
        # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model)
    model = model_adapter.model
    
    #
    

    def reset_model_device() -> None:
        if args.distribute_model:
            # distribute model across available GPUs
            utils.distribute_model(model_adapter)
        else:
            model.to(args.device)

    
    trainloader = data_utils.get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model,
        seqlen=model.seqlen,train_valid_test='train')
   
    # # original ppl
    reset_model_device()
    if args.eval_baseline:
        logger.info('------- original ppl result -------')
        for eachppl in args.ppl_tasks:
            logger.info(f'Test for PPL task: {eachppl}')
            dataset = data_utils.get_dataset(eachppl)
            test_loader = data_utils.prepare_test_dataloader(dataset=dataset["test"], tokenizer=tokenizer, batch_size=1, seqlen=model.seqlen)
            dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
            logger.info(f'Original ppl: {dataset_ppl:.4f}')
            model.seqlen = 4096
            test_loader = data_utils.prepare_test_dataloader(dataset=dataset["test"], tokenizer=tokenizer, batch_size=1, seqlen=4096)
            dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
            logger.info(f'Original ppl(DISP): {dataset_ppl:.4f}')
            model.seqlen = 2048
            utils.cleanup_memory()
    
    # 全局剪枝率评估+获取剪枝率
    num_hidden_layers,num_heads,ffn_dim,hidden_size,head_size = utils.get_model_properties(model)
    if args.global_layer_rate == 'uniform':
        prune_rate = torch.ones(size=(num_hidden_layers, 2))*args.sparsity
    else:
        prune_rate = get_global_prune_rate(model, dataloader=trainloader, args=args, logger=logger)
   
    logger.info(prune_rate)
    logger.info(f"mean pruning ratio: {prune_rate.mean(dim=0)}")
    
    utils.consolidate_model_to_single_gpu(model, target_device='cpu')
    model.eval()
    model_adapter.prune_model(args, trainloader, prune_rate, args.device, logger) 

    ## save model
    if args.save_dir:
        pass
    ## evaluation
    logger.info('################################ evaluation result #####################################')
    
    utils.distribute_model(model_adapter)
    logger.info('------- ppl result After Pruning -------')
    ppl = {}
    for eachppl in args.ppl_tasks:
        logger.info(f'Test for PPL task: {eachppl}')
        dataset = data_utils.get_dataset(eachppl)
        test_loader = data_utils.prepare_test_dataloader(dataset=dataset["test"], tokenizer=tokenizer, batch_size=1, seqlen=model.seqlen)
        dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logger.info(f'After Pruning ppl: {dataset_ppl:.4f}')
        model.seqlen = 4096
        test_loader = data_utils.prepare_test_dataloader(dataset=dataset["test"], tokenizer=tokenizer, batch_size=1, seqlen=4096)
        dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
        logger.info(f'After Pruning ppl(DISP): {dataset_ppl:.4f}')
        model.seqlen = 2048
        utils.cleanup_memory()
        
   
    logger.info('------- zero shot task result -------')
    if args.eval_zero_shot_task:
       eval_uitls.eval_zero_shot_task(model, tokenizer, args.tasks, logger)
    
    # print('################################ zero shot task result #####################################')
    # sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
    # sliced_fraction = 1.0 - sliced_param_count / original_param_count
    # logging.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')

if __name__ == "__main__":
    slicing_args = slicing_arg_parser()
    my_logger = utils.configure_logging(slicing_args, log_to_console=True, log_to_file=True, level=logging.INFO)
    slicing_main(slicing_args, my_logger)

