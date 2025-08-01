#!/bin/bash
set -e

seed=42
epochs=20
device=0
dataset_key="T15"
teacher_model="Qwen_2.5_VL_72B"

CUDA_VISIBLE_DEVICES=${device} python scripts/assistant_rd.py \
    --dataset_key ${dataset_key} \
    --batch_size 2 \
    --test_batch_size 16 \
    --epoch ${epochs} \
    --lr 0.0003 \
    --model_max_length 1024\
    --generate_max_length 512\
    --lr_patience 2\
    --lr_factor 0.5\
    --min_lr 1e-6\
    --gradient_accumulation_steps 20\
    --teacher_model ${teacher_model} \
    --seed ${seed} | tee logs/step1_${dataset_key}_${teacher_model}.txt