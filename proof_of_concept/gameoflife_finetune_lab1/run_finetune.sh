#!/bin/sh

LLAMACPP_FINETUNE="/Users/neoneye/nobackup/git/llama.cpp/finetune"
MODEL_BASE="/Users/neoneye/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
TRAIN_DATA="train_data.txt"

"${LLAMACPP_FINETUNE}" \
    --model-base "${MODEL_BASE}" \
    --train-data "${TRAIN_DATA}" \
    --lora-out lora.bin \
    --ctx 2048 \
    --sample-start "<s>" \
    --seed 18341
