
# Docker Setup
This project uses multiple Dockerfiles, each designed for a specific stage of the MLOps workflow: training, inference, evaluation, backend API serving, frontend UI, and an end-to-end data-drift demo.

All containers are reproducible and lockfile-driven, primarily using uv with uv.lock and pyproject.toml.

| Dockerfile                  | Purpose                         | Typical Usage              |
|----------------------------|---------------------------------|----------------------------|
| `backend.dockerfile`       | FastAPI backend (API server)    | Local & cloud inference    |
| `frontend.dockerfile`      | Streamlit UI                    | Local & Cloud Run          |
| `train.dockerfile`         | Model training                  | Local / CI / cloud jobs    |
| `evaluation.dockerfile`    | Model evaluation                | Reproducible metrics       |
| `inference.dockerfile`     | Batch inference                 | Heatmaps & overlays        |
| `data_drift_demo.dockerfile` | End-to-end drift demo         | Experimental pipeline      |

!!! note 
Since these files are not named `Dockerfile`, you must always use `-f` when building.


## 1. Backend API (FastAPI) — `backend.dockerfile`

### Build & run

```bash
docker build -f backend.dockerfile -t anomaly-backend .
docker run --rm -p 8000:8000 anomaly-backend
```
Entrypoint

```bash
uv run uvicorn src.anomaly_detection.api:app --host 0.0.0.0 --port 8000
```

## 2. Backend API (Bento) 

We experimented with a BentoML backend but ultimately settled on FastAPI for simplicity and consistency with the rest of the stack.

## 3. Frontend UI (Streamlit)  — `frontend.dockerfile`

Runs a Streamlit web interface for uploading images and calling the backend API. Designed to work both locally and on Google Cloud Run (uses $PORT).

### Build & run (local)

```bash
docker build -f frontend.dockerfile -t anomaly-frontend .
docker run --rm -p 8080:8080 -e PORT=8080 anomaly-frontend
```

## 4. Training — `train.dockerfile`
Runs the training pipeline in a fully reproducible container.

### Build & run
```bash
docker build -f train.dockerfile -t anomaly-train .
docker run --rm anomaly-train
```

## 5. Evaluation  — `evaluation.dockerfile`
Runs model evaluation on a dataset to compute metrics in a controlled and reproducible environment.

### Build & run
```bash
docker build -f evaluation.dockerfile -t anomaly-eval .
docker run --rm anomaly-eval
```

## 6. Inference — `inference.dockerfile`
Runs offline inference tasks such as heatmap generation, overlay visualization and batch prediction.

### Build & run
```bash
docker build -f inference.dockerfile -t anomaly-infer .
docker run --rm anomaly-infer
```

## 7. Data drifting Demo — `data_drift_demo.dockerfile`
Runs an end-to-end data drift experiment at container runtime via an entrypoint script (not during image build). Typical steps include generating drifted datasets, running the API, calling it, and producing plots.

Building and running the docker for data drifting is shown in section Pipeline/ Data Drift.

### Build & run
```bash
docker build -f data_drift_demo.dockerfile -t mlops-data-drift-demo .
```
```bash
docker run --rm `
  -v "$((Get-Location).Path)\data:/app/data" `
  -v "$((Get-Location).Path)\models:/app/models" `
  -v "$((Get-Location).Path)\results:/app/results" `
  -e COLOR_CONTRAST=0.6 `
  -e CLASS_NAME=carpet `
  -e DATA_ROOT=/app/data `
  -e OUT_ROOT=/app/results/data_drift `
  mlops-data-drift-demo

```
