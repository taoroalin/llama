# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple, Optional
import os
import sys
import torch
import torch.optim as optim
import fire
import time
import json
import wandb
from tqdm import tqdm

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel(rank_offset:int=0) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank+rank_offset)

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
    return model, tokenizer,local_rank,params


def train(
    ckpt_dir: str,
    tokenizer_path: str,
    train_json_file: str,
    lr: float = 1e-5, # they used 1.5e-4 cosine decaying to 1.5e-5
    max_seq_len: int = 2048, # 2048,
    max_batch_size: int = 32,
    epochs=1,
    skip_initial_node=False,
    save_dir=None,
    min_seq_len=None,
    truncate_seq_len:bool = False,
    wandb_name:str = "llama-finetune",
    max_dps:Optional[int]=None,
    start_gpu_num:int=0,
):
    local_rank, world_size = setup_model_parallel(start_gpu_num)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    print("loading json")
    train_json = json.load(open(train_json_file))[:max_dps]
    print("loaded json")
    # print(len(train_json), train_json[0])
    model, tokenizer,local_rank ,params= load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    string_length_limit = max_seq_len*5
    optimizer = optim.SGD(model.parameters(), lr, momentum=0)
    if skip_initial_node:
        endstd = "<|END_STANDARD_PROMPT|>"
        train_json_processed = [{**x,"prompt":endstd.join( x["prompt"].split(endstd)[1:]) if endstd in x["prompt"] else x["prompt"]} for x in train_json]
    else:
        train_json_processed = train_json
    token_pairs = [
        (
            tokenizer.encode(x["prompt"][:string_length_limit], bos=True, eos=False),
            tokenizer.encode(x["completion"][:string_length_limit], bos=False, eos=True),
        )
        for x in train_json_processed
    ]
    data_and_prefixes = [(torch.tensor([x[0] + x[1]], dtype=torch.int64).cuda(),len(x[0])) for x in token_pairs]
    if truncate_seq_len:
        data_and_prefixes_fitting = [(x[0][:,:max_seq_len],x[1]) for x in data_and_prefixes]
    else:
        data_and_prefixes_fitting = [x for x in data_and_prefixes if( x[0].shape[1]<=max_seq_len) and (min_seq_len is None or x[0].shape[1]>=min_seq_len)]
        n_within = len(data_and_prefixes_fitting)
        n_too_long = len(data_and_prefixes)-len(data_and_prefixes_fitting)
        print(f"{n_within} within size {n_too_long} too long {n_within/(n_within+n_too_long)}")
    
    if local_rank==0:
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "dataset": train_json_file,
                "dataset_size": len(data_and_prefixes_fitting),
                "initial_model_path": ckpt_dir,
                "epochs": epochs,
            },
        )
    for epoch in range(epochs):
        for i,(dp, pre) in enumerate(tqdm(data_and_prefixes_fitting)):
            optimizer.zero_grad()
            # try:
            out = model.forward_train_mode(dp, 0)
            # except:
            #     from llama import mem_report
            #     mem_report.mem_report()
            #     break
            loss = compute_loss(out, dp, pre)
            if local_rank==0:
                wandb.log({"loss": loss.detach().cpu().item(),"epoch":epoch,"len":dp.shape[1]})
            loss.backward()
            optimizer.step()
    if save_dir is not None:
        save_model(model,save_dir,local_rank,params)
        
def save_model(model,folder,local_rank,params):
        os.system(f"mkdir -p {folder}")
        if local_rank==0:
            json.dump(params,open(f"{folder}/params.json","w"))
        torch.save(model.state_dict(), f"{folder}/consolidated.0{local_rank}.pth")
        
    

def compute_loss(out, tokens, mask_pre):
    tokens = tokens[:, mask_pre:]
    out = out[:, mask_pre - 1 : -1]
    logprobs = torch.log_softmax(out, dim=-1)
    on_correct = torch.gather(logprobs, 2, tokens.unsqueeze(-1))
    loss = on_correct.mean()
    return -loss


if __name__ == "__main__":
    fire.Fire(train)

# torchrun --nproc_per_node 1 train.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/7B --tokenizer_path /home/taoroalin/llama/downloaded-weights/tokenizer.model --train_json_file=arc-data/completion_evals.json
# torchrun --nproc_per_node 8 train.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/65B --tokenizer_path /home/taoroalin/llama/downloaded-weights/tokenizer.model --train_json_file=arc-data/completion_evals.json

# to generate
# torchrun --nproc_per_node 1 example.py --ckpt_dir checkpoints/ft01 --tokenizer_path /home/taoroalin/llama/downloaded-weights/tokenizer.model --train_json_file=arc-data/completion_evals.json

# finetune for long seq
# torchrun --nproc_per_node 1 train.py --ckpt_dir /home/taoroalin/llama/downloaded-weights/7B --tokenizer_path /home/taoroalin/llama/downloaded-weights//tokenizer.model --train_json_file=pretrain_data/books.json --save-dir=checkpoints/7blong0 --epochs=1 --max-seq-len=4096 --max-dps=500 --truncate-seq-len=True --wandb-name=long-context