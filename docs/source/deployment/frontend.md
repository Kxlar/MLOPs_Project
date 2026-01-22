




# Frontend (Streamlit UI)

This project includes a lightweight Streamlit frontend that allows users to
upload an image and run anomaly detection through the backend API.

The frontend does not perform inference itself — it acts as a client that
sends images to the FastAPI backend and displays the returned prediction.

---

## Overview

The frontend provides:

- Image upload (`.png`, `.jpg`, `.jpeg`)
- One-click inference
- Visualization of:
  - Uploaded image
  - Anomaly score
  - Prediction result
- Clear separation between **UI (frontend)** and **inference (backend)**

### Architecture:

```text
User Browser
     ↓
Streamlit Frontend (port 8501)
     ↓  HTTP POST /predict
FastAPI Backend (port 8000)
     ↓
DINOv3 + Memory Bank

```

---

## Prerequisites

Before running the frontend, make sure:

1. You have built a **memory bank** using `train.py`
2. The **FastAPI backend is running**
3. Dependencies are installed via `uv`

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

If the backend is not running, you will see a connection error.

---
