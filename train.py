# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

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


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer,params


def train(
    ckpt_dir: str,
    tokenizer_path: str,
    train_json_file: str,
    lr: float = 1e-5,
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
    
    # random.shuffle(train_json)
    
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
    
    # wandb init
    if local_rank==0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="llama-7b-finetune",
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": ckpt_dir,
                "dataset": "completion_evals.json",
                "dataset_size": len(data_and_mask_filtered),
                "epochs": epochs,
            },
        )
    for i,(dp, pre) in enumerate(data_and_mask_filtered):
        ln = dp.shape[1]
        optimizer.zero_grad()
        out = model.forward_train_mode(dp, 0)
        loss = compute_loss(out, dp, pre)
        loss.backward()
        if warmup_steps is not None and i<warmup_steps:
            optimizer.lr=lr*(i+1)/(warmup_steps+1)
        else:
            optimizer.lr=math.cos(i/len(data_and_mask_filtered)*math.pi/2)
        optimizer.step()
        if local_rank==0:
            wandb.log({"loss": loss.detach().cpu().item(),"len":ln,"pre":pre})
    save_model(model,save_dir,local_rank,params)

def compute_loss(out, tokens, mask_pre):
    tokens = tokens[:, mask_pre:]
    out = out[:, mask_pre - 1 : -1]
    logprobs = torch.log_softmax(out, dim=-1)
    on_correct = torch.gather(logprobs, 2, tokens.unsqueeze(-1))
    loss = on_correct.mean()
    return -loss

def save_model(model,folder,local_rank,params):
    os.system(f"mkdir -p {folder}")
    if local_rank==0:
        json.dump(params,open(f"{folder}/params.json","w"))
    torch.save(model.state_dict(), f"{folder}/consolidated.0{local_rank}.pth")

if __name__ == "__main__":
    fire.Fire(train)

# torchrun --nproc_per_node 4 train.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/30B --tokenizer_path /home/taoroalin/llama/downloaded-weights//tokenizer.model --train_json_file=arc-data/completion_evals.json --save_dir="checkpoints/30bft0"
