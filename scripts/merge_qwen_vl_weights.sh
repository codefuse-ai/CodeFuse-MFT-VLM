python llava/merge_pretrain_weights_to_qwenvl.py \
    --LLM-path $PRETRAINED_MODEL_PATH \
    --mm-projector-type cross_attn \
    --mm-projector $PRETRAINED_MODEL_PATH/mm_projector/mm_projector.bin \
    --vision-tower $PRETRAINED_MODEL_PATH/Qwen-VL-visual/ \
    --output-path /path/to/save/model
