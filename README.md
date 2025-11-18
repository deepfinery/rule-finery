# 🧠 AML–LLM Hybrid Simulator
Train a Small Llama Model to Emulate a Retail Banking Anti–Money–Laundering (AML) Rule Engine  

---

## 📘 Overview
This project demonstrates how to **convert a symbolic rule engine (Drools)** into a **trainable dataset** for a **small language model (LLM)**.  
We simulate retail banking anti–money–laundering decisions — `CLEAR`, `REVIEW`, `SAR`, `BLOCK` — based on structured transaction data.  
Then we fine-tune a **Llama-3 3B** model with **QLoRA adapters** to imitate those rules in natural-language inference form.

---

## 🧩 Architecture
```text
Drools Rules (.drl)
   ↓
Java Runner (fires rules → JSON output)
   ↓
Python Generator (produces 100k+ labeled cases)
   ↓
QLoRA Fine-tuning (3B Llama model on 1× H100)
   ↓
LLM “Rule Engine” (predicts AML decision + escalation)
```

---

## 📂 Directory Layout
```
aml-llm/
├── data-gen/
│   ├── dataset/
│   ├── make_tx_aml_dataset.py
│   ├── quality_check.py
│   └── split_dataset.py
├── rule-engine/             # Drools runner, rules, and HTTP service
│   ├── drool-runner/
│   ├── rules/
│   └── drools_service.py
├── training/
│   ├── cloud-training-script/   # Vertex AI helper scripts + Docker image
│   ├── common/                  # Shared trainers (LoRA, QLoRA, full fine-tune)
│   └── local-training-scripts/  # Sample datasets + helper shells
└── serve/                       # vLLM deployment assets
```

---

## ⚙️ Prerequisites
- Java 17+
- Maven 3.9+
- Python 3.10+
- NVIDIA H100/A100/RTX3090+ (CUDA 12)
- Hugging Face access to [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

---

## 🏗️ 1. Build the Drools Runner
```bash
cd rule-engine/drool-runner
mvn -q -DskipTests package
# → target/drools-runner-1.0.0-shaded.jar
```
> The dataset generator (`data-gen/make_tx_aml_dataset.py`) shells out to this jar for each synthetic case. Build it once before creating new datasets. See `rule-engine/README.md` for details on the accompanying HTTP service used during benchmarking.

Smoke-test by piping any transaction JSON (or point to a file):
```bash
cat <<'JSON' | java -jar target/drools-runner-1.0.0-shaded.jar ../rules/tx_aml.drl -
{"person":{"person_id":"P1","segment":"retail","pep":false,"sanctions_hit":false,"home_country":"US","kyc_verified":true,"avg_tx_amount_90d":320,"avg_tx_per_day_90d":1.2},"account":{"account_id":"A1","opened_days_ago":400},"recent_tx":[{"tx_id":"T1","timestamp":"2025-11-12T10:42:00Z","amount":4200,"currency":"USD","direction":"out","channel":"wire","counterparty_country":"CA","counterparty_id":"CP10","merchant_category":null}]}
JSON
```

---

## 🧮 2. Generate Dataset
```bash
cd data-gen
python make_tx_aml_dataset.py 50000 dataset/tx_aml_dataset.jsonl
python split_dataset.py dataset/tx_aml_dataset.jsonl
```
Expected sizes: 80 % train | 10 % val | 10 % test.

---

## 🧠 3. Fine-Tune the Model
```bash
pip install -r training/local-training-scripts/requirements.txt
python training/common/train.py \
  --method qlora \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --dataset_file data-gen/dataset/tx_aml_dataset.jsonl \
  --output_dir training/local-training-scripts/my-finetuned-model
```

`--method` supports `qlora`, `lora`, or `full` (no adapters). QLoRA loads the base model in 4-bit with `bitsandbytes`, LoRA keeps the base model in bf16/fp16 while training adapters, and `full` updates every parameter (requires much larger GPUs). Adjust the remaining CLI flags (batch size, epochs, etc.) as needed.

---

## 📊 4. Evaluate
Metrics:
- Decision accuracy (CLEAR / REVIEW / SAR / BLOCK)
- Escalation-level exact match
- Reasons F1 (multilabel)
- Rule-coverage completeness  
Use the Drools engine as the ground truth.

---

## 🤖 5. Inference Prompt Example
```
You are an AML rule engine simulator.
Given the facts (JSON), output ONLY this JSON:
{"aml_decision":"CLEAR|REVIEW|SAR|BLOCK","reasons":[...],"escalation_level":0-3}

Facts:
{...}
```
Expected LLM response:
```json
{"aml_decision":"REVIEW","reasons":["LARGE_WIRE","ODD_HOUR"],"escalation_level":2}
```

---

## 📈 Recommended Training Scale
| Dataset Size | Expected Accuracy | GPU Time (1× H100, 2 epochs) |
|--------------:|-----------------:|------------------------------:|
| 20 k  | 85–90 % | ~1 h |
| 100 k | 95 % + | ~4 h |
| 250 k | 97 % + | ~8 h |

Tip: aim for balanced coverage of every rule and edge case rather than sheer volume.

---

## 🧪 LoRA vs QLoRA
| Method | VRAM | Speed | Accuracy (typical) | When to use |
|--------|------|--------|--------------------|-------------|
| **QLoRA** | 🟢 Low (4-bit) | ⚡ Fast | ≈ LoRA | Default |
| **LoRA**  | 🟠 Higher (bf16) | Moderate | +0–1 % | Max fidelity |
| **Full SFT** | 🔴 High | Slow | +0–1 % | Research-only |

---

## 🧱 Next Steps
- Expand DRL rules with your own AML thresholds.  
- Add **hard-example mining** (misclassified cases).  
- Test zero-shot transfer to unseen rule combos.  
- Integrate with Drools validator for hybrid guardrails.

---

## Training
```
export PROJECT_ID=hopeful-subject-478116-v6
export REGION=us-central1
export BUCKET=finery-training
```

### Vertex AI workflow (GPU fine-tuning)
The repo ships with `training/cloud-training-script/submit_vertex_job.sh`, which automates dataset upload, container build, and Vertex A100 submission:

1. **Prepare**  
   ```bash
   gcloud auth login
   gcloud config set project $PROJECT_ID
   gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
2. **Run the helper**  
   ```bash
   cd training/cloud-training-script
   ./submit_vertex_job.sh
   ```
   - Uploads `data-gen/dataset/tx_aml_dataset.jsonl` to `gs://$BUCKET/data/...`
   - Builds `training/cloud-training-script/Dockerfile` via Cloud Build using `training/cloud-training-script/cloudbuild.yaml`
   - Pushes the trainer image to Artifact Registry (`.../finery-repo/aml-llm-trainer:<timestamp>` and `:latest`)
   - Starts an A2 (A100) Vertex Custom Job with your dataset + hyperparameters.
3. **Monitor**  
   ```bash
   gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION
   ```

#### Faster incremental builds
- A top-level `.dockerignore` keeps the build context small (excludes `.git`, `.venv`, datasets, etc.).
- `training/cloud-training-script/cloudbuild.yaml` now **pulls the previous `:latest` trainer image** and passes it to `docker build --cache-from ...`, so dependency layers (CUDA, pip installs) are reused. Only changed source files trigger rebuilds.
- The script automatically retags each successful build as `:latest` for future caching. To force a cold rebuild, override the cache tag:  
  ```bash
  CACHE_IMAGE_URI=us-central1-docker.pkg.dev/$PROJECT_ID/finery-repo/aml-llm-trainer:force \
  ./submit_vertex_job.sh
  ```
- Artifact & bucket IAM bindings (Storage + Artifact Registry writer roles) are applied automatically to the Compute, Cloud Build, and Vertex service accounts on each run.

> Need to reinstall dependencies inside the container? Run `INSTALL_DEPS=1 ./submit_vertex_job.sh`.

## ⚖️ License
MIT License.  
Ruleset is illustrative and **not** a production AML policy.

---

## 👨‍💻 Author & Credits
Built with assistance from **ChatGPT (GPT-5)** and **Red Hat OpenShift AI** best practices.  
Use freely for research, prototyping, or internal model-evaluation projects.
# Rule-Generator-AML
