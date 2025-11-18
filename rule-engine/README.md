# Rule Engine Toolkit

This folder centralizes the Drools artifacts used for both dataset generation and benchmarking.

## Layout
```
rule-engine/
├── drool-runner/     # Maven project that wraps Drools rules and exposes JSON I/O
├── rules/            # .drl files (tx_aml.drl)
└── drools_service.py # FastAPI wrapper to expose Drools over HTTP for benchmarks
```

## Build the Drools runner
```
cd rule-engine/drool-runner
mvn -q -DskipTests package
# → target/drools-runner-1.0.0-shaded.jar
```

The dataset generator (`data-gen/make_tx_aml_dataset.py`) shells out to this jar for every synthetic case, so make sure it is built before running dataset scripts.

## Run the HTTP service (benchmark mode)
```
python rule-engine/drools_service.py \
  --jar rule-engine/drool-runner/target/drools-runner-1.0.0-shaded.jar \
  --rules rule-engine/rules/tx_aml.drl \
  --port 9000
```

This FastAPI app is lightweight (just wraps the jar via `subprocess`) and is meant to support benchmarking comparisons with the LLM service. The benchmark helpers under `benchmark/` assume the endpoint is reachable at `/v1/drools/score`.
