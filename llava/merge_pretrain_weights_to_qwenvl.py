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
import argparse
from llava.model import *
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class VisionArguments:
    vision_tower: Optional[str] = field(default="/mnt/project/LLAVA/Qwen-VL-visual/")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_type: Optional[str] = field(default="linear")
    pretrain_mm_mlp_adapter: Optional[str] = field(default="")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLM-path", type=str, default=None)
    parser.add_argument("--mm-projector-type", type=str, default='linear')
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="./Qwen-VL")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_path = args.LLM_path
    projector_type = args.mm_projector_type
    projector_path = args.mm_projector
    vision_tower_path = args.vision_tower

    vision_args = VisionArguments()
    vision_args.mm_projector_type = projector_type
    vision_args.vision_tower = vision_tower_path
    vision_args.pretrain_mm_mlp_adapter = projector_path

    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = LlavaQWenForCausalLM.from_pretrained(
        model_path,
        config=cfg_pretrained,
    )
    model.get_model().initialize_vision_modules(vision_args)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    output_dir = args.output_path
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer._save(output_dir)

