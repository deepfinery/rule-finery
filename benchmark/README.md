# Benchmark Toolkit

This folder holds utilities to evaluate the AML LoRA service and the Drools ground-truth engine.

## Dependencies
Install the helper stack into a local virtualenv:

```bash
pip install -r benchmark/requirements.txt
```

## Benchmark the LLM service
Run the OpenAI-compatible endpoint benchmark and export a text summary:

```bash
BASE_URL=http://18.118.130.132:8000 \
MODEL=aml-qlora \
benchmark/run_llm_benchmark.sh data-gen/dataset/tx_aml_dataset.jsonl reports/llm.txt --limit 2000
```

The helper script also writes a JSON summary next to the text file (override with `JSON_OUTPUT=/path/to/llm.json`) that you can feed into the comparison tooling below.

## Benchmark the Drools service
First expose the Drools runner jar via FastAPI:

```bash
python rule-engine/drools_service.py \
  --jar rule-engine/drool-runner/target/drools-runner-1.0.0-shaded.jar \
  --rules rule-engine/rules/tx_aml.drl \
  --port 9000
```

Then point the HTTP benchmarker at it:

```bash
python benchmark/benchmark_drools.py \
  --dataset data-gen/dataset/tx_aml_dataset.jsonl \
  --base-url http://localhost:9000 \
  --summary-json reports/drools.json
```

## Compare outputs
Generate JSON summaries for both engines and feed them to the comparator:

```bash
python benchmark/compare_benchmarks.py reports/llm.json reports/drools.json
```

The script prints a simple table (accuracy, latency, throughput, exact match) so you can quantify the LLM's fidelity relative to Drools.
