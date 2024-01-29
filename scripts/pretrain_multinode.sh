#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
#MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=qwen
########### DO NOT CHANGE ###########

accelerate launch \
    --num_machines 8 \
    --num_processes 64 \
    --use_deepspeed \
    --deepspeed_multinode_launcher 'standard' \
    --zero_stage 2 \
    --offload_optimizer_device 'cpu' \
    --offload_param_device 'none' \
    --gradient_accumulation_steps 1 \
    --gradient_clipping 1.0 \
    --zero3_init_flag false \
    --zero3_save_16bit_model false \
    --main_training_function 'main' \
    --mixed_precision 'bf16' \
    --dynamo_backend 'no' \
    --same_network \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend 'static' \
    llava/train/train_mem.py \
    --model_name_or_path /path/to/model/ \
    --version $PROMPT_VERSION \
    --data_path /path/to/train/data/ \
    --eval_path /path/to/eval/data/ \
    --is_parquet True \
    --image_folder /path/to/images/ \
    --image_aspect_ratio None \
    --vision_tower /path/to/visiontower/ \
    --mm_projector_type cross_attn \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /path/to/save/weights/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
