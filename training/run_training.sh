#!/bin/bash
python3 train_qlora.py \
    --dataset_file "./data/tx_aml_dataset.jsonl" \
    --output_dir "./my-finetuned-model"
