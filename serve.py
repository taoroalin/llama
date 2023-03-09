# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from train_util import *
api_keys = {"maGpAFLfHJpyBkYQL43ia3r5JRAfH9Bxo2W464"}

max_seq_len = 2048
tokenizer_path =  "/home/taoroalin/llama/downloaded-weights/tokenizer.model"
model_path =  "/home/taoroalin_gmail_com/llama/checkpoints/65Bdistill0/steps1500"

local_rank, world_size = setup_model_parallel()

model,tokenizer,_=load(
        model_path, tokenizer_path, local_rank, world_size, max_seq_len, 32
    )
generator = LLaMA(model, tokenizer)

import torch.distributed as tdist

def completions_per_proc(data):

    prompts= data["prompt"] if isinstance(data["prompt"],list) else [data["prompt"]]
    if data["n"]>1:
        prompts = prompts*data["n"]
    results = generator.generate(
        prompts, max_gen_len=data["max_tokens"], temperature=data["temp"], top_p=0.95
    )
    return[ {"completion":x} for x in results]
object_list = [None]

if local_rank==0:
    from flask import Flask, request,jsonify,abort

    app = Flask(__name__)
    print("rank 0!")
    @app.route("/completions", methods=["POST"])
    def completions():
        print("got request")
        data = request.json
        if data["api_key"] not in api_keys:
            raise Exception("Unauthorized")
        object_list[0]=data
        tdist.broadcast_object_list(object_list,src=0)
        result = {"outputs":completions_per_proc(data)}
        print("result",result)
        return jsonify(result)
    app.run('0.0.0.0', 5000,debug=True,threaded=False)
else:
    while True:
        print("waiting for data")
        # if local_rank==0:
        #     object_list[0] = json.loads(input())
        tdist.broadcast_object_list(object_list,src=0)
        data = object_list[0]
        print(f"{local_rank} got {data}")
        result = completions_per_proc(data)
        if local_rank==0:
            print(result)