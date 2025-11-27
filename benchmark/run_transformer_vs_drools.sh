#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAMPLES="${SAMPLES:-1000}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/training/transformer_training/local_artifacts/transformer_multihead.pt}"
DROOLS_JAR="${DROOLS_JAR:-$REPO_ROOT/rule-engine/drool-runner/target/drools-runner-1.0.0-shaded.jar}"
DROOLS_RULES="${DROOLS_RULES:-$REPO_ROOT/rule-engine/rules/tx_aml.drl}"
REASON_THRESHOLD="${REASON_THRESHOLD:-0.5}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_ROOT/benchmark/transformer_vs_drools.json}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary '$PYTHON_BIN' not found. Set PYTHON_BIN to your interpreter." >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$DROOLS_JAR" ]]; then
  echo "Drools jar not found at $DROOLS_JAR, building..." >&2
  (cd "$REPO_ROOT/rule-engine/drool-runner" && mvn -q -DskipTests package)
fi

if [[ ! -f "$DROOLS_JAR" ]]; then
  echo "Failed to locate Drools jar at $DROOLS_JAR after build." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Running transformer vs Drools benchmark..."
"$PYTHON_BIN" "$REPO_ROOT/benchmark/transformer_vs_drools.py" \
  --samples "$SAMPLES" \
  --checkpoint "$CHECKPOINT" \
  --drools-jar "$DROOLS_JAR" \
  --drools-rules "$DROOLS_RULES" \
  --reason-threshold "$REASON_THRESHOLD" \
  --device "$DEVICE" \
  --output-json "$OUTPUT_JSON"

echo "Benchmark results saved to $OUTPUT_JSON"
