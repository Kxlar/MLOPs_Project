
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

For BentoML app to do inference:

1. Initialize bentoml
```bash
uv run bentoml --version
```
2. Convert the torch model into a ONNX model
```bash
uv run export_onnx_paranoid.py
```
3. Build the bento container
```bash
uv run bentoml build
```
4. Launch docker in the background. bBuild the docker container based on the bento - replace TAG by the number that shows up after step 3
```bash
bentoml containerize anomaly_detection_service:<TAG>
```
5. Run the container
```bash
docker run --rm -p 3000:3000 anomaly_detection_service:<TAG>
```

6. To use the backend, go to http://localhost:3000 

OR (once the docker image is running) in new terminal run:
```bash
uv run src/anomaly_detection/api_inference.py
```

## 3. Frontend UI (Streamlit)  — `frontend.dockerfile`

This project includes a lightweight Streamlit frontend that allows users to
upload an image and run anomaly detection through the backend API.

The frontend does not perform inference itself — it acts as a client that
sends images to the FastAPI backend and displays the returned prediction.


---

## 1. Running the Backend (Required)

The frontend expects the backend to be available at:
```bash
http://127.0.0.1:8000
```

Start the backend in a dedicated terminal and keep it running:

```bash
cd ~/projects/MLOPs_Project
uv run uvicorn src.anomaly_detection.api:app --host 0.0.0.0 --port 8000
```

## 2. Running the Frontend (Streamlit)

Open a second terminal and run:

```bash
cd ~/projects/MLOPs_Project
uv run streamlit run frontend/frontend.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

You should see:

You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501


Open in your browser:
```bash
http://localhost:8501

```

---

### Using the Frontend

1. Open the Streamlit page

2. Upload an image from the MVTec dataset (or a compatible image)

3. Click Run prediction

The frontend will:

* Send the image to /predict

* Wait for the backend response

* Display prediction results

---


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
