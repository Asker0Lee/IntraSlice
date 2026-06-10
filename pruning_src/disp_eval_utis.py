

'''
DISP-LLM, LLM-Surgeon PPL test function.!!!!
That is diffierent to SliceGPT

[1] Gao, S.; Lin, C.-H.; Hua, T.; Tang, Z.; Shen, Y.; Jin, H.; and Hsu, Y.-C. 2024. Disp-llm: Dimension-independent structural pruning for large language models. Advances in Neural Information Processing Systems, 37: 72219–72244

'''


import math

import torch
from torch import nn


@torch.no_grad()
def eval_model_ppl_with_LLM_Surgeon_TestFunction(model, model_str, enc, dev="cuda"):
    '''
    Eval PPL with LLM_Surgeon
    '''
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    enc = enc.input_ids

    enc = enc.to(dev)
    nsamples = enc.numel() // model.seqlen

    losses = 0.0
    for i in range(nsamples):
        if (i % 10) == 0:
            print(f"\tPass {i+1} of {nsamples}")
        batch = enc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)

        out = model(batch)

        if "logits" in out.keys():
            logits = out["logits"]
        else:
            raise ValueError(f"Unknown model out keys:", out.keys())

        loss = nn.CrossEntropyLoss()
        L = loss(logits[:, :-1, :].view(-1, logits.size(-1)), batch[:, 1:].view(-1))

        losses += L.item()

    ppl = math.exp(losses / nsamples)

    model.config.use_cache = use_cache

    outdir = {}
    outdir["ppl"] = ppl

    return outdir


