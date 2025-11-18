#!/usr/bin/env bash

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required. See serve/README.md for installation steps." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_URI="${1:-${MODEL_GCS_URI:-}}"
DEST_DIR="${ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}"

if [[ -z "$MODEL_URI" ]]; then
  echo "Usage: $(basename "$0") gs://bucket/path/to/model_folder" >&2
  echo "       MODEL_GCS_URI env var is also accepted." >&2
  exit 1
fi

if [[ "${MODEL_URI}" != gs://* ]]; then
  echo "MODEL URI must start with gs://. Got: ${MODEL_URI}" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "Syncing artifacts..."
echo "  Source: $MODEL_URI"
echo "  Dest:   $DEST_DIR"
gcloud storage rsync "$MODEL_URI" "$DEST_DIR" --recursive --delete-unmatched-destination=false

echo "Done. Adapter files are under $DEST_DIR."
