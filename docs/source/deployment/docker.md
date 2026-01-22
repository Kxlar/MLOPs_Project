This project uses multiple Dockerfiles, each designed for a specific stage of the MLOps workflow: training, inference, evaluation, backend API serving, frontend UI, and an end-to-end data-drift demo.

All containers are reproducible and lockfile-driven, primarily using uv with uv.lock and pyproject.toml.


## 1. FastAPI backend Ducker

## 2. Bento Ducker

comments to build the docker

## 2. Streamlit frontend UI Ducker

we tried with both backends and we settled for the FastAPI.

## Testing Docker implementation running the model
Model evaluation container
train.dockerfile

Build & run
docker build -f train.dockerfile -t anomaly-train:latest .
docker run --rm anomaly-train:latest


inference.dockerfile
evaluation.dockerfile

## 2. Data drifting Ducker