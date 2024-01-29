#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
#MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 llava/train/train_mem.py \
    --model_name_or_path /mnt/user/laiyan/salesgpt/model/Qwen-14B-Chat \
    --deepspeed ./scripts/new_ds_config.json \
    --version $PROMPT_VERSION \
    --data_path /mnt/project/LLAVA/image2schema_data/dataset_img2ir_v2/dataset/converted/yuque_5w_zh/datasets/ \
    --eval_path /mnt/project/LLAVA/image2schema_data/dataset_img2ir_v2/dataset/converted/yuque_5w_zh/datasets/ir.json \
    --image_folder /mnt/project/LLAVA/image2schema_data/dataset_img2ir_v2/dataset/converted/yuque_5w_zh/train/ \
    --vision_tower /mnt/project/LLAVA/cn_clip_336/ \
    --mm_projector_type mlp3x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /mnt/project/LLAVA/pretrain_weights/llava_qwen-14b-multigpu-10xlr/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-2 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
