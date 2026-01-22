## 5. Data Preparation
# Week 1 — Foundations

| Requirement | Where | Evidence |
|---|---|---|
| Dedicated environment (uv) | `pyproject.toml`, `uv.lock` | `uv sync` works |
| Cookiecutter structure | repo layout | `src/`, `tests/`, `docs/` |
| Data pipeline | `src/anomaly_detection/data.py` | preprocessing command |
| Training procedure | `src/anomaly_detection/train.py` | produces `models/memory_bank.pt` |
| Dockerfiles | `dockerfiles/` | `docker build ...` |
| Config files | `configs/` | `train.yaml`, `evaluate.yaml` |
| Profiling | `evaluate.prof`, notes | command used |
| Logging | (wherever used) | log outputs |

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
