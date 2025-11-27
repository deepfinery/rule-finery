#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOCAL_DATASET_PATH="${LOCAL_DATASET_PATH:-$REPO_ROOT/data-gen/dataset/tx_aml_dataset.jsonl}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$REPO_ROOT/training/transformer_training/local_artifacts}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}"
LOCAL_LR="${LOCAL_LR:-3e-4}"
LOCAL_DEVICE="${LOCAL_DEVICE:-cuda}"
LOCAL_IMAGE_NAME="${LOCAL_IMAGE_NAME:-aml-transformer-local}"
LOCAL_USE_GPU="${LOCAL_USE_GPU:-1}"
MAX_TRANSACTIONS="${MAX_TRANSACTIONS:-20}"

if [[ ! -f "$LOCAL_DATASET_PATH" ]]; then
  echo "Dataset not found at $LOCAL_DATASET_PATH" >&2
  exit 1
fi

if [[ "$LOCAL_DATASET_PATH" != "$REPO_ROOT"* ]]; then
  echo "LOCAL_DATASET_PATH must reside under the repo root ($REPO_ROOT) so it can be mounted into the container." >&2
  exit 1
fi

if [[ "$LOCAL_OUTPUT_DIR" != "$REPO_ROOT"* ]]; then
  echo "LOCAL_OUTPUT_DIR must reside under the repo root ($REPO_ROOT) so it can be mounted into the container." >&2
  exit 1
fi

mkdir -p "$LOCAL_OUTPUT_DIR"

DATASET_CONTAINER_PATH="/app${LOCAL_DATASET_PATH#$REPO_ROOT}"
OUTPUT_CONTAINER_PATH="/app${LOCAL_OUTPUT_DIR#$REPO_ROOT}"

echo "Building local transformer trainer image ($LOCAL_IMAGE_NAME)..."
docker build -t "$LOCAL_IMAGE_NAME" -f "$REPO_ROOT/training/cloud-training-script/Dockerfile" "$REPO_ROOT"

GPU_FLAG=()
if [[ "$LOCAL_USE_GPU" == "1" ]]; then
  GPU_FLAG=(--gpus all)
fi

echo "Running local transformer training (epochs=$LOCAL_EPOCHS)..."
docker run --rm "${GPU_FLAG[@]}" \
  -v "$REPO_ROOT":/app \
  -w /app \
  -e TRAIN_ENTRY_MODULE=training.transformer_training.train \
  "$LOCAL_IMAGE_NAME" \
    --dataset_file "$DATASET_CONTAINER_PATH" \
    --output_dir "$OUTPUT_CONTAINER_PATH" \
    --epochs "$LOCAL_EPOCHS" \
    --batch_size "$LOCAL_BATCH_SIZE" \
    --learning_rate "$LOCAL_LR" \
    --max_transactions "$MAX_TRANSACTIONS" \
    --device "$LOCAL_DEVICE"

echo "Local training complete. Artifacts stored in $LOCAL_OUTPUT_DIR"
