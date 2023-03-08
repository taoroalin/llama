# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from train_util import *
from typing import Tuple,Optional
import os
import sys
import torch
import torch.optim as optim
import fire
import time
import json
import wandb
import random
import math

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def train(
    ckpt_dir: str,
    tokenizer_path: str,
    train_json_file: str,
    lr: float = 3e-5,
    max_seq_len: int = 2048,
    max_batch_size: int = 32,
    truncate_seq_len:bool=False,
    epochs=1,
    save_dir="checkpoints/13bft0",
    warmup_steps=20,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    train_json = json.load(open(train_json_file))
    
    same_perm_start_len = 50
    perm = torch.randperm(len(train_json)-same_perm_start_len)
    new_train_json = train_json[:same_perm_start_len]
    for i,p in enumerate(perm):
        new_train_json.append(train_json[p+same_perm_start_len])
    train_json=new_train_json
    
    if truncate_seq_len:
        for i,x in enumerate(train_json):
            x["prompt"] = x["prompt"][:max_seq_len*5]
            x["completion"] = x["completion"][:max_seq_len*5]
            
    print(len(train_json), train_json[0])
    model, tokenizer ,params= load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    optimizer = optim.SGD(model.parameters(), lr, momentum=0)
    token_pairs = [
        (
            tokenizer.encode(x["prompt"], bos=True, eos=False),
            tokenizer.encode(x["completion"], bos=False, eos=True),
        )
        for x in train_json
    ]
    data_and_mask = [(torch.tensor([x[0] + x[1]], dtype=torch.int64).cuda(),len(x[0])) for x in token_pairs]
    if truncate_seq_len:
        data_and_mask = [(x[0][:,:max_seq_len],x[1]) for x in data_and_mask]
    data_and_mask_filtered = [x for x in data_and_mask if x[0].shape[1]<=max_seq_len]
    
    if local_rank==0:
        wandb.init(
            project="llama-7b-finetune",
            config={
                "learning_rate": lr,
                "architecture": ckpt_dir,
                "dataset": "completion_evals.json",
                "dataset_size": len(data_and_mask_filtered),
                "epochs": epochs,
            },
        )
    try:
        for i,(dp, pre) in enumerate(data_and_mask_filtered):
            stime=time.time()
            ln = dp.shape[1]
            optimizer.zero_grad()
            out = model.forward_train_mode(dp, 0)
            loss,loss_pos = compute_loss(out, dp, pre)
            loss.backward()
            if warmup_steps is not None and i<warmup_steps:
                optimizer.lr=lr*(i+1)/(warmup_steps+1)
            else:
                optimizer.lr=lr*math.cos(i/(len(data_and_mask_filtered)*1.2)*math.pi/2)
            optimizer.step()
            if local_rank==0:
                wandb.log({"loss": loss.detach().cpu().item(),"len":ln,"pre":pre,"lr":optimizer.lr,"interval":time.time()-stime, **{f"loss_pos{i}":x for i,x in enumerate(loss_pos) if (i-1)%800==0}})
            if i!=0 and i%500==0:
                save_model(model,save_dir+f"/steps{i}",local_rank,params)
    except Exception as e:
        save_model(model,save_dir,local_rank,params)
        raise e
    save_model(model,save_dir,local_rank,params)

def compute_loss(out, tokens, mask_pre):
    tokens = tokens[:, mask_pre:]
    out = out[:, mask_pre - 1 : -1]
    logprobs = torch.log_softmax(out, dim=-1)
    on_correct = torch.gather(logprobs, 2, tokens.unsqueeze(-1))
    on_correct_by_pos = on_correct[0,:,0].detach().cpu()
    loss = on_correct.mean()
    return -loss, -on_correct_by_pos

if __name__ == "__main__":
    fire.Fire(train)

# torchrun --nproc_per_node 2 train.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/13B --tokenizer_path /home/taoroalin/llama/downloaded-weights//tokenizer.model --train_json_file=pretrain_data/books_16000_5000.json --save_dir="checkpoints/13Blong1" --truncate-seq-len=True
