#!/bin/bash
set -e

dataset_key="T15"
device=0
teacher_model="Qwen_2.5_VL_72B"

CUDA_VISIBLE_DEVICES=${device} python scripts/assistant_augment.py \
    --mode diverse_reasoning \
    --dataset_key ${dataset_key} \
    --batch_size 16 \
    --n_diversity 1 \
    --temperature 0.7 \
    --model_max_length 1024 \
    --generate_max_length 512 \
    --teacher_model ${teacher_model} \
    --ft_cot_lr 0.0003 | tee logs/step2_${dataset_key}_${teacher_model}.txt