#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL_ID="${BASE_MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-$SCRIPT_DIR/artifacts}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$BASE_MODEL_ID}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TP_SIZE="${TP_SIZE:-1}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
VLLM_DEVICE="${VLLM_DEVICE:-gpu}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-false}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/vllm_server.log}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "Adapter directory not found: $ADAPTER_DIR" >&2
  echo "Run serve/download_artifacts.sh first or set ADAPTER_DIR." >&2
  exit 1
fi

if [[ -z "$HF_TOKEN" ]]; then
  echo "Hugging Face token is required. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN." >&2
  exit 1
fi

export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

CMD=(python -m vllm.entrypoints.openai.api_server
  --model "$BASE_MODEL_ID"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --tensor-parallel-size "$TP_SIZE"
  --load-format bitsandbytes
  --quantization bitsandbytes
  --trust-remote-code
  --enable-lora
  --max-lora-rank "$MAX_LORA_RANK"
  --lora-modules "aml-qlora=${ADAPTER_DIR}"
)

if [[ "${ENFORCE_EAGER,,}" == "true" ]]; then
  CMD+=(--enforce-eager)
fi

if [[ "$VLLM_DEVICE" == "cpu" ]]; then
  CMD+=(--device cpu --enforce-eager)
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($VLLM_EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "Starting vLLM with:"
echo "  Base model:   $BASE_MODEL_ID"
echo "  Adapter dir:  $ADAPTER_DIR"
echo "  Host:port:    $HOST:$PORT"
echo "  HuggingFace:  token set"
echo "  TP size:      $TP_SIZE"
echo "  Device mode:  $VLLM_DEVICE"
echo "  Extra args:   ${VLLM_EXTRA_ARGS:-<none>}"

if [[ "${RUN_IN_BACKGROUND,,}" == "true" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo "vLLM server started in background (PID $!). Logs: $LOG_FILE"
else
  exec "${CMD[@]}"
fi
