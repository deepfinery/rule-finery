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
├── rules/
│   └── tx_aml.drl
├── drools-runner/
│   ├── pom.xml
│   └── src/main/java/demo/Runner.java
├── data/
│   ├── make_tx_aml_dataset.py
│   ├── split_dataset.py
│   └── tx_aml_dataset.jsonl
└── training/
    ├── requirements.txt
    ├── map_facts_to_text.py
    └── train_qlora.py
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
cd drools-runner
mvn -q -DskipTests package
# → target/drools-runner-1.0.0-shaded.jar
```

Test once:
```bash
java -jar target/drools-runner-1.0.0-shaded.jar ../rules/tx_aml.drl ../data/sample.json
```

---

## 🧮 2. Generate Dataset
```bash
cd data
python make_tx_aml_dataset.py 50000 tx_aml_dataset.jsonl
python split_dataset.py tx_aml_dataset.jsonl
```
Expected sizes: 80 % train | 10 % val | 10 % test.

---

## 🧠 3. Fine-Tune the Model (QLoRA)
```bash
cd training
pip install -r requirements.txt
export MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
accelerate launch train_qlora.py
```

This trains a 4-bit QLoRA adapter (~200 MB) on your dataset using 1× H100.

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

## ⚖️ License
MIT License.  
Ruleset is illustrative and **not** a production AML policy.

---

## 👨‍💻 Author & Credits
Built with assistance from **ChatGPT (GPT-5)** and **Red Hat OpenShift AI** best practices.  
Use freely for research, prototyping, or internal model-evaluation projects.
# Rule-Generator-AML
