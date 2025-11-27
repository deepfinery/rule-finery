# Structured Transformer Serving

This folder packages a lightweight FastAPI app for the multi-head transformer
trained in `training/transformer_training/`.

## 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r serve/serve-transformer/requirements.txt
```

## 2. Launch the API
```bash
python serve/serve-transformer/serve_transformer.py \
  --checkpoint training/transformer_training/artifacts/transformer_multihead.pt \
  --host 0.0.0.0 \
  --port 9000
```

## 3. Sample request
```bash
curl -s http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"facts": {"person": {"person_id": "P6514"}}, "top_k_reasons": 3}' | jq .
```

The response includes `aml_decision`, `escalation_level`, and the highest-scoring reasons.
