# Getting Started

This guide helps you run the anomaly detection project **locally** as quickly as possible.
It assumes no prior knowledge of the codebase.

---


## 1. System Requirements

- Linux / macOS / Windows  
  *(Windows users should use WSL)*
- Python ≥ 3.9 (3.13 recommended)
- Docker
- Git


---

## 2. Installation
Clone the repository:
```bash
git clone <REPO_URL>
cd MLOPs_Project
```

## 3. Data and Pre-trained model
Downloads the data and the pretrained model from GCP bucket

```bash
chmod +x setup.sh
```
```bash
./setup.sh
```


Install dependencies:
```bash
uv sync
```
Verify installation:
```bash
uv run python -c "import torch; print('OK')"
```




## 3. Project Structure
```text
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


```
## Invoke tasks
```bash
uv run invoke --list
```


---

## 4. Minimal End-to-End Run

### 4.1 Data loading: 
The project expects the MVTec AD dataset in the following structure:


```bash title="data.py"
uv run python src/anomaly_detection/data.py \
  --data_root ./data/raw \
  --class_name carpet \
  --img_size 224 \
  --batch_size 8
```

### 4.2 Build the memory bank:
```bash title="train.py"
uv run python src/anomaly_detection/train.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --save_path ./models/memory_bank.pt \
  --img_size 224 \
  --batch_size 8
```

This step:
- extracts features
- builds a memory bank
- saves model artifacts under models/

### 4.3 Run inference:
```bash title="inference.py"
uv run python src/anomaly_detection/inference.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./reports/figures \
  --img_size 224 \
  --batch_size 8 \
  --k 10
```

Outputs include:
- anomaly heatmaps
- anomaly scores

### 4.4 Evaluate the model:
```bash title="evaluate.py"
uv run python src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./reports/figures/eval \
  --img_size 224 \
  --batch_size 8 \
  --k 10
```

Evaluation outputs are written to:
- results/

---

## 5. Run the API

To start the FastAPI backend locally:
```bash
uv run python src/anomaly_detection/api.py
```

Open in browser:
```bash
http://localhost:8000
```

Health check:
```bash
curl http://localhost:8000/health
```

---

## 6. Run with Docker

Build and run the backend API using Docker:
```bash
docker build -f dockerfiles/backend.dockerfile -t anomaly-backend .
docker run -p 8000:8000 anomaly-backend
```


---