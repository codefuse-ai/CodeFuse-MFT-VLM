import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation import GenerationConfig
import torch
import json
import os
import os.path as osp
import shortuuid
import numpy as np
from transformers import Trainer

import shortuuid

model_base_path = "/mnt/project/LLAVA/Qwen-VL-Chat/"
cross_attn_path = "/mnt/project/LLAVA/pretrain_weights/qwen-vl-7b-yuque-box-frame-1212/checkpoint-13000/mm_projector.bin"

tokenizer = AutoTokenizer.from_pretrained(model_base_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_base_path, low_cpu_mem_usage=True, trust_remote_code=True)
cross_attn_state_dict = torch.load(cross_attn_path)

visual_state_dict = model.transformer.visual.state_dict()

def match_k(visual_state_dict, cross_attn_state_dict):
    print("number of parameters to change", len(cross_attn_state_dict))
    cnt = 0
    for k in cross_attn_state_dict:
        real_k = ".".join(k.split(".")[2:])
        if real_k in visual_state_dict:
            visual_state_dict[real_k] =  cross_attn_state_dict[k]
            cnt += 1
    print("number of parameters changed", cnt)
    return visual_state_dict

visual_state_dict = match_k(visual_state_dict, cross_attn_state_dict)
model.transformer.visual.load_state_dict(visual_state_dict)

output_dir = "/mnt/project/LLAVA/pretrain_weights/merged/qwen-vl-7b-yuque-box-frame-1212/"
trainer = Trainer(model=model, tokenizer=tokenizer)
trainer._save(output_dir)

