# 🧠 AML–LLM Hybrid Simulator

Train a compact Llama-3 model to emulate a Drools-based retail banking AML engine. This repo takes symbolic rules, generates synthetic labeled data, fine-tunes a QLoRA adapter, deploys it with vLLM, and benchmarks the model against the original rule engine.

---

## 🚀 Project Goals
1. **Codify business rules** in Drools (`rule-engine/`) and treat them as the ground truth.
2. **Mass-produce training data** (100k+ JSON cases) via `data-gen/`.
3. **Fine-tune a small Llama model** using lightweight Hugging Face/PEFT tooling (`training/`).
4. **Serve the adapter** with vLLM behind an OpenAI-compatible API (`serve/`).
5. **Benchmark & compare** the LLM vs the rule engine (accuracy, latency, throughput) using scripts in `benchmark/`.

---

## 📁 Repository Layout
```
aml-llm/
├── benchmark/                # Benchmarking, comparison, Drools vs LLM tooling
├── data-gen/                 # Synthetic data generator and dataset utilities
├── rule-engine/              # Drools runner, rules, and HTTP service wrapper
├── serve/                    # vLLM runtime scripts + deployment notes
├── training/                 # Fine-tuning scripts (local + Vertex AI helpers)
└── README.md                 # This file
```

---

## 🔧 Prerequisites
* Java 17+, Maven 3.9+
* Python 3.10+
* CUDA-capable GPU (L4/A10/A100/H100) for training + serving
* Hugging Face access to [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* (Optional) Google Cloud CLI for Vertex AI workflows

Environment variables commonly used:
```bash
export HF_TOKEN=hf_xxx                        # Hugging Face access token
export BASE_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
```

---

## 📚 Quickstart
1. **Build the Drools runner**
   ```bash
   cd rule-engine/drool-runner
   mvn -q -DskipTests package            # produces target/drools-runner-1.0.0-shaded.jar
   ```
2. **Generate a dataset**
   ```bash
   cd data-gen
   python make_tx_aml_dataset.py 50000 dataset/tx_aml_dataset.jsonl
   python split_dataset.py dataset/tx_aml_dataset.jsonl
   ```
3. **Fine-tune locally (QLoRA)**
   ```bash
   pip install -r training/local-training-scripts/requirements.txt
   python training/common/train.py \
     --method qlora \
     --model_id $BASE_MODEL_ID \
     --dataset_file data-gen/dataset/tx_aml_dataset.jsonl \
     --output_dir training/local-training-scripts/my-finetuned-model
   ```
4. **Serve with vLLM**
   ```bash
   pip install -r serve/requirements.txt
   export HF_TOKEN=hf_xxx
   export ADAPTER_DIR=/opt/aml-llm/serve/artifacts   # downloaded adapter path
   ./serve/run_vllm_server.sh                        # exposes http://0.0.0.0:8000/v1
   ```
5. **Benchmark the endpoint**
   ```bash
   pip install -r benchmark/requirements.txt
   BASE_URL=http://<llm-host>:8000 MODEL=aml-qlora \
     benchmark/run_llm_benchmark.sh \
       data-gen/dataset/tx_aml_dataset.jsonl \
       reports/llm.txt \
       --limit 2000
   ```

---

## 🧩 End-to-End Workflow

### 1. Rule Engine (`rule-engine/`)
* **`drool-runner/`** – Maven project that wraps Drools rules and accepts JSON I/O.
* **`rules/tx_aml.drl`** – Canonical AML rules.
* **`drools_service.py`** – FastAPI wrapper to expose the runner at `/v1/drools/score` for benchmarking.

> Build the runner _once_ before generating data or benchmarking:
> ```bash
> cd rule-engine/drool-runner && mvn -q -DskipTests package
> ```

### 2. Data Generation (`data-gen/`)
* **`make_tx_aml_dataset.py`** – Creates balanced scenarios (CLEAR/REVIEW/SAR/BLOCK), shells out to the Drools jar for ground-truth labels, and writes JSONL plus audit files.
* **`split_dataset.py`** – 80/10/10 train/val/test split.
* **`quality_check.py`** – Coverage diagnostics (reasons, fired rules, decision balance).

Key knobs inside `make_tx_aml_dataset.py`:
* Scenario priors (`SCENARIO_PROBS`) and class ratios (`TARGET_RATIOS`).
* KYC profiles, transaction generators, and reason deduping.

### 3. Training (`training/`)
* **`common/train.py`** – Unified CLI with `--method` = `qlora`, `lora`, or `full`.
* **`local-training-scripts/`** – Requirements + examples for single-GPU runs.
* **`cloud-training-script/`** – Vertex AI helper (`submit_vertex_job.sh`) that builds the trainer image, uploads data to GCS, and launches A100/H100 jobs.

Typical local run:
```bash
python training/common/train.py \
  --method qlora \
  --model_id $BASE_MODEL_ID \
  --dataset_file data-gen/dataset/tx_aml_dataset.jsonl \
  --output_dir my-finetuned-model \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4
```

Vertex workflow highlights:
1. `gcloud auth login` + `gcloud config set project ...`
2. `cd training/cloud-training-script && ./submit_vertex_job.sh`
3. Monitor with `gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION`

### 4. Serving (`serve/`)
* **`download_artifacts.sh`** – Sync PEFT adapters from GCS to the VM.
* **`run_vllm_server.sh`** – Launches `vllm.entrypoints.openai.api_server` with bitsandbytes, eager mode, and the LoRA adapter registered as `aml-qlora`.
  * Environment knobs: `ADAPTER_DIR`, `PORT`, `HOST`, `MAX_LORA_RANK`, `RUN_IN_BACKGROUND`, etc.
  * Endpoint: `http://HOST:PORT/v1/chat/completions` (OpenAI-compatible). Call with `model: "aml-qlora"` to apply the adapter.

Health check example:
```bash
curl -s http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "aml-qlora",
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [
          {"role": "system", "content": "You are an AML rule engine simulator. Return JSON only."},
          {"role": "user", "content": "Facts: {\"amount\": 54000, \"channel\": \"WIRE\", \"hour\": 2, \"country\": \"DE\"}"}
        ]
      }'
```

### 5. Benchmarking & Comparison (`benchmark/`)
* **`benchmark_endpoint.py`** – Core evaluator for the LLM endpoint (accuracy, reasons F1, latency, throughput). Accepts `--summary-json` for downstream tooling.
* **`run_llm_benchmark.sh`** – Convenience wrapper that writes both human-readable (`.txt`) and machine-readable (`.json`) summaries.
* **`benchmark_drools.py`** – Replays the dataset against the Drools HTTP service to capture baseline metrics.
* **`drools_service.py`** (in `rule-engine/`) – FastAPI proxy so the rule engine can be benchmarked like any other HTTP service.
* **`compare_benchmarks.py`** – Consumes the JSON summaries and prints a table (decision accuracy, avg latency, throughput, exact match).

Workflow:
```bash
# 1) Run LLM benchmark
BASE_URL=http://<llm-host>:8000 MODEL=aml-qlora \
  benchmark/run_llm_benchmark.sh data-gen/dataset/tx_aml_dataset.jsonl reports/llm.txt --limit 2000

# 2) Serve Drools and benchmark it
python rule-engine/drools_service.py --port 9000
python benchmark/benchmark_drools.py \
  --dataset data-gen/dataset/tx_aml_dataset.jsonl \
  --base-url http://localhost:9000 \
  --summary-json reports/drools.json

# 3) Compare
python benchmark/compare_benchmarks.py reports/llm.json reports/drools.json
```

---

## 🗺️ Architecture Summary
```text
rule-engine/rules (Drools)
   ↓  mvn package
rule-engine/drool-runner (shaded jar)
   ↓  subprocess
data-gen/make_tx_aml_dataset.py (JSONL + audit)
   ↓
training/common/train.py (LoRA/QLoRA/full)
   ↓
serve/run_vllm_server.sh (vLLM OpenAI API, LoRA adapter)
   ↓
benchmark/{benchmark_endpoint, benchmark_drools}.py + compare
```

---

## 🧭 Roadmap & Ideas
* Expand rule coverage (odd channels, FX, sanctions lists) and regenerate data.
* Integrate hard-example mining (misclassified cases) back into Drools to tighten guardrails.
* Wire benchmarks into CI/CD or scheduled cron to catch regressions.
* Explore larger adapters (rank > 32) or full SFT once hardware permits.

---

## 🤝 Contributions
1. Fork the repo (`https://github.com/deepfinery/rule-finery`), create a topic branch, and submit a PR.
2. Please keep instructions up to date when moving files or changing scripts (e.g., new rule-engine paths).
3. Open issues for bugs, missing docs, or ideas—we’re building the AML simulator in the open.

Happy hacking! 🛡️
