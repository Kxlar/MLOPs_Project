## Documentation

Documentation for anomaly_detection

# Project usage guide

This document explains how to install, run, and use the anomaly detection project.
It is written for **new users** who want to reproduce training, inference, evaluation, or run the system as a service.

---

## 1. Overview

This project implements an **image anomaly detection pipeline** using deep feature
extraction and memory-bank–based scoring. It supports:

- Local Python execution
- Hydra-based configurable pipelines
- Batch execution via Docker
- Running as an API service

---

## 2. Requirements

### System
- Linux / macOS / Windows (WSL recommended on Windows)
- Python 3.13
- Docker

### Python environment
This project uses **uv** for dependency management.

Install uv (if not already installed):
pip install uv

## 3. Installation
Clone the repository:
git clone <REPO_URL>
cd MLOPs_Project


uv sync
uv run invoke --list
uv run python -c "import anomaly_detection; print('OK')"

Install dependencies:
uv sync

Verify installation:
uv run python -c "import torch; print('OK')"

## 4. Project Structure
src/anomaly_detection/
├── api.py                # FastAPI entrypoint
├── service.py            # Backend service logic
├── train.py              # Training
├── inference.py          # Inference
├── evaluate.py           # Evaluation
├── hydra/
│   ├── train_hydra.py
│   ├── inference_hydra.py
│   ├── evaluate_hydra.py
│   ├── augment_hydra.py
│   └── hydra_utils.py
configs/
├── train.yaml
├── inference.yaml
├── evaluate.yaml
├── augment.yaml
dockerfiles/
├── train.dockerfile
├── inference.dockerfile
├── evaluation.dockerfile
├── backend.dockerfile
├── frontend.dockerfile


## Invoke tasks
uv run invoke --list
uv run invoke preprocess-data
uv run invoke train
uv run invoke test

## Data preprocessing

This project separates **raw data** from **processed data** to ensure
reproducibility and prevent accidental data corruption.


## 5. Data Preparation
data/
└── carpet/
    ├── train/
    ├── test/

## 6. Train
uv run python src/anomaly_detection/train.py \
  --data_root ./data \
  --class_name carpet

## 6.1 Training with Hydra
uv run python src/anomaly_detection/hydra/train_hydra.py \
  data_root=./data \
  class_name=carpet


## 7. Inference
uv run python src/anomaly_detection/inference.py \
  --data_root ./data \
  --class_name carpet

## 7.1 Inference with Hydra
uv run python src/anomaly_detection/hydra/inference_hydra.py \
  data_root=./data \
  class_name=carpet

## Outputs:
# anomaly heatmaps
# anomaly scores

## 8. Evaluation
uv run python src/anomaly_detection/evaluate.py \
  --k 5 \
  --output_dir ./results

## 8.1 Evaluation with Hydra
uv run python src/anomaly_detection/hydra/evaluate_hydra.py \
  k=5 \
  output_dir=./results


## 9. Running as an API (FastAPI)
uv run python src/anomaly_detection/api.py

# Default URL: http://localhost:8000
# Example endpoint: curl http://localhost:8000/health

## 10. Docker Usage
# 10.1 Training:
docker build -f dockerfiles/train.dockerfile -t anomaly-train .
docker run --rm anomaly-train

# 10.2 Inference
docker build -f dockerfiles/inference.dockerfile -t anomaly-infer .
docker run --rm anomaly-infer

# 10.3 Evaluation
docker build -f dockerfiles/evaluation.dockerfile -t anomaly-eval .
docker run --rm anomaly-eval

# 10.4 Backend API
docker build -f dockerfiles/backend.dockerfile -t anomaly-backend .
docker run -p 8000:8000 anomaly-backend

## Tests
uv run pytest


## Outputs:
# - Hydra outputs are stored in outputs/ (ignored files)


# Simulates 10 users sending requests
locust -f ./tests/test_api_load_perf.py --headless -u 10 -r 1 -t 1m --host http://localhost:8000

