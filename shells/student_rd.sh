#!/bin/bash
set -e

teacher_device=0
device=1
seed=42
epochs=20
dataset_key="T15"
teacher_model="Qwen_2.5_VL_72B"

CUDA_VISIBLE_DEVICES=${teacher_device},${device} python scripts/student_rd.py \
    --training_mode vanilla \
    --dataset_key $dataset_key \
    --n_aug_diversity 1 \
    --teacher_model_size 7B \
    --student_model_size 3B \
    --batch_size 2 \
    --test_batch_size 24 \
    --epoch ${epochs} \
    --seed 42 \
    --kd_temperature 2.0 \
    --kd_lambda 0.3 \
    --model_max_length 1024\
    --generate_max_length 512\
    --lr_patience 2\
    --lr_factor 0.5\
    --min_lr 1e-6\
    --gradient_accumulation_steps 20\
    --ft_cot_lr 0.0003 \
    --teacher_model ${teacher_model} \
    --lr 0.0003 | tee logs/step3_${dataset_key}_${teacher_model}.txt