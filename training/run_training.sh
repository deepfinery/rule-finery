#!/bin/bash
set -euo pipefail

python3 train_qlora.py \
    --dataset_file "./data/tx_aml_dataset.jsonl" \
    --output_dir "./my-finetuned-model" \
    --max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --eval_steps 500 \
    --save_steps 500
