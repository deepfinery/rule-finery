# Training API

This lightweight Express service exposes the fine-tuning script (`training/common/train.py`) as a REST API so DeepFinery Studio (Next.js front-end + Node.js backend) can launch training runs programmatically. The API routes training requests to different backends:

- **Kubernetes** (default) – submits a `batch/v1 Job` that runs the trainer image with the provided CLI flags.
- **Vertex AI** – creates a CustomJob for teams that still need the managed GCP path.
- **Local process** – keeps the old `python train.py ...` flow for local development only.

---
## Components you need
- **Training API** (this package) – HTTP entrypoint.
- **Trainer image** (for k8s/Vertex) – container with `training/common/train.py` as its entrypoint (see `training/cloud-training-script/Dockerfile` for a starting point).
- **Kubernetes cluster with GPUs** – provision via Terraform samples in `training/infra/terraform/{eks,gke,aks,coreweave}`; ensures an A100/H100-capable node pool and emits a kubeconfig.
- **Optional Vertex AI** – if you want managed GCP runs instead of Kubernetes.
- **Storage bucket** (GCS or S3) – holds datasets and model outputs; the API assumes it already exists and credentials are handled externally.

---
## Setup

```bash
cd training/training-api
npm install
npm start   # listens on PORT (default 4000)
```

Core environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `4000` | HTTP port. |
| `MAX_LOG_CHARS` | `1000000` | Rolling log tail limit. |
| `DEFAULT_TRAINING_BACKEND` | `kubernetes` | `kubernetes`, `vertex`, or `local`. |
| `PYTHON_BIN` | `python3` | Local backend only. |
| `TRAIN_SCRIPT_PATH` | `training/common/train.py` | Local backend only. |

Kubernetes backend:

| Variable | Default | Purpose |
| --- | --- | --- |
| `K8S_TRAINING_IMAGE` | _(required)_ | Trainer image URI (entrypoint should run `train.py`). |
| `K8S_NAMESPACE` | `default` | Namespace for the Job. |
| `K8S_SERVICE_ACCOUNT` | _empty_ | Service account for bucket access. |
| `K8S_APPLY` | `0` | When `1`, the API will run `kubectl apply -f -`. Otherwise it prints the manifest only. |
| `KUBECTL_BIN` | `kubectl` | Override kubectl path. |
| `K8S_KUBECONFIG` | _empty_ | Path to kubeconfig (defaults to current context). |
| `K8S_CONTEXT` | _empty_ | Optional kubeconfig context to target. |
| `K8S_NODE_SELECTOR` | _empty_ | JSON map for node selector. |
| `K8S_TOLERATIONS` | _empty_ | JSON array of tolerations. |
| `K8S_IMAGE_PULL_SECRETS` | _empty_ | JSON array of image pull secret names/objects. |
| `K8S_BACKOFF_LIMIT` | `0` | Job backoff limit. |
| `K8S_TTL_SECONDS_AFTER_FINISHED` | _unset_ | TTL for completed jobs. |

Vertex AI backend:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VERTEX_PROJECT` / `PROJECT_ID` | _(required)_ | GCP project id. |
| `VERTEX_REGION` | `us-central1` | Region for the CustomJob. |
| `VERTEX_IMAGE_URI` | _(required)_ | Trainer container image. |
| `VERTEX_MACHINE_TYPE` | `a2-highgpu-1g` | Machine type. |
| `VERTEX_ACCELERATOR_TYPE` | `NVIDIA_TESLA_A100` | GPU type. |
| `VERTEX_ACCELERATOR_COUNT` | `1` | GPU count. |
| `VERTEX_REPLICA_COUNT` | `1` | Worker replicas. |
| `VERTEX_SERVICE_ACCOUNT` | _empty_ | Optional custom service account. |
| `VERTEX_NETWORK` | _empty_ | Optional VPC network. |
| `VERTEX_SUBMIT` | `0` | When `1`, the API will call `gcloud ai custom-jobs create`. Otherwise it prints the spec only. |
| `GCLOUD_BIN` | `gcloud` | Override gcloud path. |

---
## Kick-start the platform (Kubernetes path)
1) **Provision a GPU cluster**: choose a Terraform recipe under `training/infra/terraform`:
   - `eks` (AWS), `gke` (GCP), `aks` (Azure), or `coreweave` (CKS). Run `terraform init && terraform apply` with provider vars. After apply, export `K8S_KUBECONFIG=$(pwd)/kubeconfig` (and `K8S_CONTEXT` if needed).
2) **Build/publish the trainer image** containing `train.py` (or reuse an existing one). Set `K8S_TRAINING_IMAGE` to that image and `K8S_APPLY=1` to auto-apply Jobs.
3) **Run the Training API**:
   - Local: `npm start` with the env vars above.
   - Docker: `docker build -t training-api -f training/training-api/Dockerfile .` then:
     ```bash
     docker run --rm -p 4000:4000 \
       -e K8S_TRAINING_IMAGE=... \
       -e K8S_APPLY=1 \
       -e K8S_KUBECONFIG=/kube/config \
       -v ~/.kube/config:/kube/config:ro \
       training-api
     ```
   - Docker Compose: from `training/training-api`, run `docker compose up --build -d` and edit `docker-compose.yml` envs as needed.
4) **Call the API**: POST to `/train` (schema below) with your dataset/model params. The API will `kubectl apply` the Job manifest against the kubeconfig/context supplied. For Vertex, set `backend=vertex` and provide the `VERTEX_*` envs instead.

Access needed:
- Kubernetes: kubeconfig credentials that allow creating `batch/v1` Jobs in the target namespace and pulling the trainer image.
- Vertex: gcloud/service account able to run `gcloud ai custom-jobs create` and access the dataset/output buckets.
- Storage: read access to `dataset_file` URI and write access to `upload_output_to` (if set). IAM specifics are handled outside this repo.

---
## Endpoints

### `GET /health`
Returns a simple status plus the detected backends.

### `POST /train`
Launches a training job on the requested backend (default `kubernetes`). Request shape:

- `backend` (optional): `kubernetes`, `vertex`, or `local`.
- `job_name` (optional): human-readable label; falls back to a UUID.
- `dataset_file` **required**: for Kubernetes/Vertex this must be a `gs://` or `s3://` URI that the pod/job service account can read.
- `output_dir` **required**: local path inside the container/VM. Use `upload_output_to` (gs://) if you need the trained model copied to cloud storage.
- `model_id` or `model`/`model_name`: base model to pull from Hugging Face or another hub.
- `trainer_args`: any extra `train.py` flags (e.g., `num_train_epochs`, `lora_rank`, `gradient_checkpointing`, etc.).
- `env`: extra environment variables passed to the backend (including `HF_TOKEN` if omitted from the payload).

Examples:

**Kubernetes (default)**
```json
{
  "backend": "kubernetes",
  "job_name": "aml-qlora-k8s",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "dataset_file": "s3://my-bucket/jobs/job-123/data.jsonl",
  "output_dir": "/tmp/model",
  "upload_output_to": "gs://finery-training/models/job-123",
  "trainer_args": {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8
  },
  "env": {
    "HF_TOKEN": "hf_xxx"
  }
}
```

**Vertex AI fallback**
```json
{
  "backend": "vertex",
  "job_name": "aml-qlora-vertex",
  "model_id": "nemotron-3-8b",
  "dataset_file": "gs://finery-training/jobs/job-456/data.jsonl",
  "output_dir": "/tmp/model",
  "upload_output_to": "gs://finery-training/models/job-456",
  "trainer_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4
  }
}
```

Notes:
- Boolean flags (e.g., `"packing": true`) automatically render as `--packing`.
- The API merges `trainer_args` with any top-level fields (aside from reserved keys like `backend`, `env`, `job_name`).
- Model aliases `model` and `model_name` map to `model_id` for convenience.
- For local backend, relative paths are resolved from the repo root (`${REPO_ROOT}`).

Response: `202 Accepted` with the queued job metadata (`id`, backend, status, normalized args, log tail, etc.).

### `GET /jobs`
Lists all known jobs with their latest status and truncated logs.

### `GET /jobs/:id`
Returns the same metadata as `/jobs` plus the full log timeline so Studio can render streaming output.

## Integration Tips
- Poll `/jobs/:id` until `status` is `succeeded` or `failed`. For Kubernetes/Vertex backends the API marks jobs as `submitted` after handing off to the platform; a controller/watcher can be added later to sync real runtime phases.
- Logs are appended in-order with timestamps + stream (`stdout`/`stderr`). For Kubernetes/Vertex the API records the generated manifest/spec and any CLI output from `kubectl`/`gcloud`.
- State lives in memory; persist `jobs` if you deploy the API as a singleton or add a queue/DB for durability.
