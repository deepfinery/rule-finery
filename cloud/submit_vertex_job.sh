#!/bin/bash

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required. Install it from https://cloud.google.com/sdk/docs/install" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-hopeful-subject-478116-v6}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-aml-llm-qlora-$(date +%Y%m%d-%H%M%S)}"
BUCKET="${BUCKET:-finery-training}"
IMAGE_URI="${IMAGE_URI:-}"
CLOUDSDK_CONTAINER_BUILDING_BACKEND="${CLOUDSDK_CONTAINER_BUILDING_BACKEND:-CLOUD_BUILD}"
LOCAL_DATASET_PATH="${LOCAL_DATASET_PATH:-$REPO_ROOT/training/data/tx_aml_dataset.jsonl}"
DATASET_GCS_PATH="${DATASET_GCS_PATH:-}"
OUTPUT_GCS_PATH="${OUTPUT_GCS_PATH:-}"
BUILD_IMAGE_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_IMAGE_URI="${BUILD_IMAGE_URI:-us-central1-docker.pkg.dev/${PROJECT_ID}/finery-repo/aml-llm-trainer:${BUILD_IMAGE_TIMESTAMP}}"
CACHE_IMAGE_URI="${CACHE_IMAGE_URI:-us-central1-docker.pkg.dev/${PROJECT_ID}/finery-repo/aml-llm-trainer:latest}"
BUILD_STAGING_DIR="${BUILD_STAGING_DIR:-gs://${BUCKET}/build-artifacts}"
BUCKET_URI="gs://${BUCKET}"

if [[ -z "$PROJECT_ID" ]]; then
  echo "Set PROJECT_ID env var or configure gcloud (gcloud config set project YOUR_PROJECT)." >&2
  exit 1
fi

if [[ -z "$BUCKET" ]]; then
  echo "Set BUCKET to the target GCS bucket (without gs://)." >&2
  exit 1
fi

PROJECT_NUMBER="${PROJECT_NUMBER:-$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')}"

echo "Ensuring bucket IAM bindings for service accounts..."
for MEMBER in \
  "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  "serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  "serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
do
  gcloud storage buckets add-iam-policy-binding "$BUCKET_URI" \
    --member="$MEMBER" \
    --role="roles/storage.objectAdmin" \
    --quiet >/dev/null || true
done

echo "Ensuring Artifact Registry permissions..."
for MEMBER in \
  "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  "serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  "serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"
do
  gcloud artifacts repositories add-iam-policy-binding finery-repo \
    --location=us-central1 \
    --member="$MEMBER" \
    --role="roles/artifactregistry.writer" \
    --quiet >/dev/null || true
done

if [[ -z "$DATASET_GCS_PATH" ]]; then
  DATASET_GCS_PATH="gs://${BUCKET}/data/tx_aml_dataset.jsonl"
fi

if [[ -z "$OUTPUT_GCS_PATH" ]]; then
  OUTPUT_GCS_PATH="gs://${BUCKET}/models/${JOB_NAME}"
fi

if [[ ! -f "$LOCAL_DATASET_PATH" ]]; then
  echo "Local dataset file not found: $LOCAL_DATASET_PATH" >&2
  exit 1
fi

echo "Uploading dataset..."
echo "  Local: $LOCAL_DATASET_PATH"
echo "  GCS:   $DATASET_GCS_PATH"
gcloud storage cp "$LOCAL_DATASET_PATH" "$DATASET_GCS_PATH"

echo "Building custom training image..."
echo "  Context: $REPO_ROOT"
echo "  Tag:     $BUILD_IMAGE_URI"
gcloud builds submit "$REPO_ROOT" \
  --config="$REPO_ROOT/cloud/cloudbuild.yaml" \
  --gcs-source-staging-dir="$BUILD_STAGING_DIR" \
  --project="$PROJECT_ID" \
  --substitutions="_IMAGE_URI=${BUILD_IMAGE_URI},_CACHE_IMAGE_URI=${CACHE_IMAGE_URI}"

EFFECTIVE_IMAGE="${IMAGE_URI:-$BUILD_IMAGE_URI}"

echo "Submitting Vertex AI CustomJob:"
echo "  Project:   $PROJECT_ID"
echo "  Region:    $REGION"
echo "  Job name:  $JOB_NAME"
echo "  Dataset:   $DATASET_GCS_PATH"
echo "  Output:    $OUTPUT_GCS_PATH"
echo "  Image:     $EFFECTIVE_IMAGE"
echo "  Builder:   $CLOUDSDK_CONTAINER_BUILDING_BACKEND"

export CLOUDSDK_CONTAINER_BUILDING_BACKEND

gcloud ai custom-jobs create \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,replica-count=1,container-image-uri=${EFFECTIVE_IMAGE}" \
  --args="--dataset_file=${DATASET_GCS_PATH}","--output_dir=/tmp/model","--upload_output_to=${OUTPUT_GCS_PATH}","--max_length=2048","--packing","--per_device_train_batch_size=4","--per_device_eval_batch_size=4","--gradient_accumulation_steps=2","--learning_rate=1e-4","--num_train_epochs=3","--logging_steps=50","--eval_steps=200","--save_steps=200","--warmup_ratio=0.05","--bf16"
