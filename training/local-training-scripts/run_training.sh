#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_FILE="$REPO_ROOT/data-gen/dataset/tx_aml_dataset.jsonl"
OUTPUT_DIR="$REPO_ROOT/training/local-training-scripts/my-finetuned-model"

cd "$REPO_ROOT"

python3 training/common/train.py \
    --method qlora \
    --dataset_file "$DATASET_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 512 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --learning_rate 2e-4 \
    --logging_steps 20 \
    --eval_steps 500 \
    --save_steps 500
