# AML‑LLM Serving Playbook

This folder packages everything needed to pull the fine‑tuned QLoRA adapter from Cloud Storage, install `vLLM`, and expose the AML model behind an OpenAI‑compatible API on an Ubuntu VM (GPU or CPU fallback). The workflow assumes the adapters were uploaded by `training/common/train.py --method qlora --upload_output_to gs://...` or by the Vertex pipeline (`training/cloud-training-script/submit_vertex_job.sh`).

## Files
- `requirements.txt` – Python dependencies for runtime inference (vLLM + helpers).
- `download_artifacts.sh` – Downloads a model/artifact folder from GCS to the VM.
- `run_vllm_server.sh` – Launches the HTTP server with the base Llama model + LoRA adapter applied.
- `.gitignore` – Keeps large downloaded artifacts out of Git.

## 1. Prepare the Ubuntu host
1. **Provision hardware** – Ubuntu 22.04 LTS, NVIDIA L4/A10/A100 (or CPU-only for testing). Install NVIDIA drivers + CUDA 12.4 runtime (or newer) before continuing.
2. **System packages**
   ```bash
   sudo apt update
   sudo apt install -y python3.10 python3.10-venv python3-pip git build-essential
   ```
3. **Google Cloud CLI** (needed for `gcloud storage rsync`)
   ```bash
   curl -sSL https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud components install gcloud-storage
   gcloud auth login
   gcloud config set project hopeful-subject-478116-v6
   ```
   If you use service accounts, run `gcloud auth activate-service-account --key-file FILE.json`.
4. **Hugging Face access** – Create a token with access to `meta-llama/Llama-3.2-3B-Instruct`, then store it securely:
   ```bash
   export HF_TOKEN="hf_xxx"             # keep secret
   ```
   The scripts fall back to `HUGGING_FACE_HUB_TOKEN` if `HF_TOKEN` is unset.

## 2. Python environment + dependencies
```bash
cd /opt/aml-llm                # or wherever you cloned this repo
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch that matches your CUDA version before vLLM.
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124   # GPU path
# For CPU-only testing use: pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

pip install -r serve/requirements.txt
```

> Tip: Set `export HF_HUB_ENABLE_HF_TRANSFER=1` to use the faster hf-transfer downloader when available.

## 3. Download the fine‑tuned adapter from GCS
1. Identify the artifact path (each training run uploads to `gs://finery-training/models/<job-name>`).
   ```bash
   gcloud storage ls gs://finery-training/models/
   export MODEL_GCS_URI="gs://finery-training/models/aml-llm-qlora-20240918-153000"
   ```
2. Pull it down with the helper script (defaults to `serve/artifacts`).
   ```bash
   ./serve/download_artifacts.sh "$MODEL_GCS_URI"
   # or: ARTIFACTS_DIR=/models/aml ./serve/download_artifacts.sh "$MODEL_GCS_URI"
   ```
   The directory will contain the PEFT adapter weights (`adapter_config.json`, `adapter_model.safetensors`, tokenizer, etc.).

## 4. Start the vLLM server
```bash
export HF_TOKEN="hf_xxx"                       # if not already set
export BASE_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
export ADAPTER_DIR="serve/artifacts"           # wherever download_artifacts.sh synced
export PORT=8000
export HOST=0.0.0.0
./serve/run_vllm_server.sh
```

Environment knobs supported by `run_vllm_server.sh`:
- `BASE_MODEL_ID` – Base Hugging Face model.
- `ADAPTER_DIR` – Path to the downloaded LoRA adapter.
- `PORT` / `HOST` – Bind address.
- `MAX_MODEL_LEN` – Context window (default 2048).
- `GPU_MEM_UTIL` – Target GPU memory utilization (default 0.9).
- `TP_SIZE` – Tensor parallel degree (default 1; set to the number of GPUs).
- `VLLM_DEVICE` – Set to `cpu` for CPU-only testing (vLLM will be slow).
- `VLLM_EXTRA_ARGS` – Additional advanced flags passed directly to vLLM.

The script exports `HUGGING_FACE_HUB_TOKEN` for vLLM, enables LoRA loading, and exposes an OpenAI-compatible endpoint at `http://$HOST:$PORT/v1`.

## 5. Issue a test request
Use the OpenAI-compatible `/v1/chat/completions` route and ask vLLM to apply the adapter (`lora_name` matches the one declared in the server script).

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "lora_request": {"lora_name": "aml-qlora"},
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [
          {"role": "system", "content": "You are an AML rule engine simulator. Return JSON only."},
          {"role": "user", "content": "Facts: {\"amount\": 54000, \"channel\": \"WIRE\", \"hour\": 2, \"country\": \"DE\"}"}
        ]
      }' | jq .
```

Expect a deterministic JSON response describing the AML decision. For bare `v1/completions`, replace `messages` with `prompt`.

## 6. Operational notes
- **Autosave artifacts** under `/opt/aml-llm/serve/artifacts` to reuse across VM rebuilds. The `.gitignore` keeps these blobs out of Git.
- **CPU mode** is only for sanity checks; throughput is <1 req/s. Prefer an NVIDIA GPU (L4 gives ~70 tok/s with this model).
- **Monitoring** – The script prints Token/sec and request logs. For production, wrap the command in `systemd` or `supervisord`.
- **Upgrades** – To update weights, rerun `download_artifacts.sh` with the new `gs://` path and restart the server. To update dependencies, rerun `pip install -r serve/requirements.txt --upgrade`.

You are now ready to serve the AML simulator model anywhere that can pull from GCS and reach Hugging Face.
