# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from concurrent.futures import process
from train_util import *
from typing import Tuple, Optional
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
import itertools


def process_distill_data(
    sequences, tokenizer, max_seq_len=2048, max_dps: int = 800
) -> torch.Tensor:
    print("special tokens",tokenizer.bos_id,tokenizer.eos_id,tokenizer.pad_id)
    tokens = []
    masks = []
    n_rejected_no_model = 0
    for z, seq in enumerate(sequences):
        if all(x["type"] != "modelOutput" for x in seq):
            continue
        toks = [tokenizer.encode(x["body"], bos=False, eos=False) for x in seq]
        lens = torch.tensor([len(x) for x in toks])
        lenbefore = lens.cumsum(dim=0)
        total_len = sum(lens)
        mask = torch.zeros(total_len).long()
        alltoks = torch.tensor(list(itertools.chain(*toks))).long()
        for i, s in enumerate(seq):
            if s["type"] == "modelOutput":
                mask[lenbefore[i] : lenbefore[i] + lens[i]] = 1
        chunk_size = 1024
        n_contexts = int(total_len / chunk_size) - 1
        for i in range(n_contexts):
            dp = alltoks[i * chunk_size : (i + 2) * chunk_size]
            msk = mask[i * chunk_size : (i + 2) * chunk_size]
            dp[0] = tokenizer.bos_id
            if msk.sum() > 30:
                if len(tokens) < 5:
                    print([x["body"] for x in seq], msk)
                tokens.append(dp)
                masks.append(dp)
            else:
                n_rejected_no_model += 1
        if len(tokens) > max_dps:
            tokens = tokens[:max_dps]
            masks = masks[:max_dps]
            break
    print("rejected", n_rejected_no_model, "got", len(tokens))

    return torch.stack(tokens, 0), torch.stack(masks, 0)


def train(
    ckpt_dir: str,
    train_json_file: str,
    tokenizer_path: str = "/home/taoroalin/llama/downloaded-weights/tokenizer.model",
    lr: float = 3e-5,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    epochs=1,
    save_dir="checkpoints/13bft0",
    warmup_steps=20,
    max_dps: int = 800,
    gpu_rank_offset: int = 0,
):
    local_rank, world_size = setup_model_parallel(gpu_rank_offset)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    sequences = json.load(open(train_json_file))

    model, tokenizer, params = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, 32
    )

    tokens, masks = process_distill_data(sequences, tokenizer, max_seq_len, max_dps)
    batched_tokens, batched_masks = tokens[
        : int(tokens.shape[0] / batch_size) * batch_size
    ].reshape(-1, batch_size, tokens.shape[-1]), masks[
        : int(tokens.shape[0] / batch_size) * batch_size
    ].reshape(
        -1, batch_size, masks.shape[-1]
    )
    optimizer = optim.SGD(model.parameters(), lr, momentum=0)
    if local_rank == 0:
        wandb.init(
            project="llama-7b-context-distill",
            config={
                "learning_rate": lr,
                "architecture": ckpt_dir,
                "dataset": "completion_evals.json",
                "dataset_size": len(tokens),
                "epochs": epochs,
            },
        )
    try:
        for i, (dp, mask) in enumerate(zip(batched_tokens, batched_masks)):
            stime = time.time()
            optimizer.zero_grad()
            dp = dp.cuda()
            mask = mask.cuda()
            out = model.forward_train_mode(dp, 0)
            loss = compute_loss(out, dp, mask)
            loss.backward()
            if warmup_steps is not None and i < warmup_steps:
                optimizer.lr = lr * (i + 1) / (warmup_steps + 1)
            else:
                optimizer.lr = lr * math.cos(
                    i / (len(batched_tokens) * 1.2) * math.pi / 2
                )
            optimizer.step()
            if local_rank == 0:
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "lr": optimizer.lr,
                        "interval": time.time() - stime,
                    }
                )
            if i != 0 and i % 500 == 0:
                save_model(model, save_dir + f"/steps{i}", local_rank, params)
    except Exception as e:
        save_model(model, save_dir, local_rank, params)
        raise e
    save_model(model, save_dir, local_rank, params)


def compute_loss(out, tokens, mask):
    print("shapes", out.shape, tokens.shape, mask.shape)
    mask = mask[:, 1:]
    logprobs = torch.log_softmax(out[:, :-1], dim=-1)
    on_correct = torch.gather(logprobs, 2, tokens[:, 1:].unsqueeze(-1))[:, :, 0]
    on_correct *= mask
    loss = on_correct.mean() / mask.float().mean()
    return -loss


if __name__ == "__main__":
    fire.Fire(train)

# torchrun --nproc_per_node 1 --rdzv_backend=c10d  --rdzv_endpoint=localhost:29501 train_distill.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/7B  --train_json_file=arc-data/context-distill-sequences_pos.json --save_dir="checkpoints/7Bdistill0" --gpu-rank-offset=1
