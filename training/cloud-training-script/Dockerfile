# CUDA runtime base image with Python tooling installed
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY training/llm_training/local-training-scripts/requirements.txt /tmp/llm_requirements.txt
COPY training/transformer_training/requirements.txt /tmp/transformer_requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      -r /tmp/llm_requirements.txt \
      -r /tmp/transformer_requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu121

COPY . /app

ENTRYPOINT ["python3", "training/cloud-training-script/gcp_entry.py"]
