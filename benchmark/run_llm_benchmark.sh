#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_PATH OUTPUT_FILE [extra args for benchmark_endpoint.py]" >&2
  echo "Example: BASE_URL=http://18.118.130.132:8000 $0 data-gen/dataset/tx_aml_dataset.jsonl reports/llm.txt --limit 1000" >&2
  exit 1
fi

DATASET="$1"
OUTPUT_FILE="$2"
shift 2

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-aml-qlora}"
JSON_BASE="${OUTPUT_FILE%.*}"
if [[ "$JSON_BASE" == "$OUTPUT_FILE" ]]; then
  JSON_BASE="$OUTPUT_FILE"
fi
JSON_OUTPUT="${JSON_OUTPUT:-${JSON_BASE}.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$JSON_OUTPUT")"

python benchmark/benchmark_endpoint.py \
  --dataset "$DATASET" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --summary-json "$JSON_OUTPUT" \
  "$@"

python - "$JSON_OUTPUT" "$OUTPUT_FILE" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
lines = [
    "LLM Benchmark Report",
    f"Timestamp: {datetime.utcnow().isoformat()}Z",
    f"Dataset: {summary['dataset']}",
    f"Endpoint: {summary['endpoint']}",
    f"Model: {summary['model']}",
    f"Records evaluated: {summary['evaluated']} / {summary['total']}",
    f"Decision accuracy: {summary['decision_accuracy']:.3f}",
    f"Average latency (s): {summary['avg_latency']:.2f}",
    f"Throughput (req/s): {summary['throughput']:.2f}",
]
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote benchmark summary to {out_path}")
PY
echo "JSON summary stored at $JSON_OUTPUT"
