import sys
import io
import base64
import contextlib
import argparse
import uvicorn
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- 1. Setup Python Path ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.data import build_dataloaders
from src.anomaly_detection.model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
    upsample_anomaly_map,
    reduce_anomaly_map,
)
import torchvision.transforms as T


# --- 2. Global Configuration & State ---
class APIConfig:
    def __init__(self):
        # Default values (can be overwritten by argparse)
        self.data_root = "./data"
        self.class_name = "carpet"
        self.weights_path = "./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        self.memory_bank_path = "./models/memory_bank.pt"
        self.img_size = 224
        self.batch_size = 8
        self.host = "0.0.0.0"
        self.port = 8000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global instance to hold runtime config
config = APIConfig()
ml_models = {}


# --- 3. Auto-Build Logic ---
def run_auto_build(cfg: APIConfig, feature_extractor, device):
    """
    Automatically builds the memory bank if it is missing.
    Mimics the logic from train.py.
    """
    print(f"[-] Memory bank not found at {cfg.memory_bank_path}")
    print("[-] Starting automatic build process...")

    # Create a namespace object that looks like argparse args expected by data.py
    # We set augment=False because we are building the bank (similar to train.py)
    mock_args = SimpleNamespace(
        data_root=cfg.data_root,
        class_name=cfg.class_name,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        augment=False,
        aug_types=[],
    )

    try:
        # 1. Load Data
        print("    Loading training data...")
        # build_dataloaders returns: train_ds, test_ds, train_loader, test_loader
        _, _, train_loader, _ = build_dataloaders(mock_args)

        # 2. Build Bank
        print("    Extracting features (this may take a while)...")
        memory_bank = build_memory_bank(feature_extractor, train_loader, device)
        print(f"    Memory Bank shape: {memory_bank.shape}")

        # 3. Save
        save_path = Path(cfg.memory_bank_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(memory_bank, save_path)
        print(f"[-] Memory bank saved to {save_path}")

        return memory_bank

    except Exception as e:
        print(f"[!] Critical Error during auto-build: {e}")
        raise RuntimeError(
            "Failed to auto-build memory bank. Check dataset path."
        ) from e


# --- 4. Lifespan (Startup/Shutdown) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles model loading and optional auto-training on startup.
    """
    device = config.device
    print(f"--- Starting API for class: {config.class_name} on {device} ---")

    # 1. Load Backbone (DINOv3)
    if not Path(config.weights_path).exists():
        # We cannot auto-download proprietary/custom weights usually, so we raise error
        print(f"[!] Error: Weights file not found at {config.weights_path}")
        raise FileNotFoundError("DINOv3 weights missing.")

    dinov3 = load_dinov3(config.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3).eval().to(device)
    ml_models["feature_extractor"] = feature_extractor

    # 2. Load or Build Memory Bank
    mb_path = Path(config.memory_bank_path)

    if mb_path.exists():
        print(f"[-] Loading memory bank from {mb_path}...")
        memory_bank = torch.load(mb_path, map_location=device)
    else:
        # TRIGGER AUTO-BUILD
        memory_bank = run_auto_build(config, feature_extractor, device)

    ml_models["memory_bank"] = memory_bank
    print("--- System Ready ---")

    yield

    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- 5. API Definition ---
app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)


class PredictionResponse(BaseModel):
    filename: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    heatmap_base64: str = None


def transform_image_bytes(image_bytes: bytes, img_size: int) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(img)


def generate_heatmap_base64(
    anomaly_map_up: np.ndarray, original_img_bytes: bytes, img_size: int
) -> str:
    am_norm = (anomaly_map_up - anomaly_map_up.min()) / (
        anomaly_map_up.max() - anomaly_map_up.min() + 1e-8
    )
    orig_img = (
        Image.open(io.BytesIO(original_img_bytes))
        .convert("RGB")
        .resize((img_size, img_size))
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(orig_img)
    ax.imshow(am_norm, cmap="jet", alpha=0.5)
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- 6. Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if "memory_bank" not in ml_models:
        raise HTTPException(
            status_code=503, detail="System initializing or failed to load models."
        )

    try:
        contents = await file.read()

        # Transform
        img_t = transform_image_bytes(contents, config.img_size)

        # Inference
        feature_extractor = ml_models["feature_extractor"]
        memory_bank = ml_models["memory_bank"]

        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=10)

        # Metrics
        score = reduce_anomaly_map(anomaly_map, mode="max")
        am_up = upsample_anomaly_map(anomaly_map, config.img_size)

        # Visuals
        heatmap_b64 = generate_heatmap_base64(am_up, contents, config.img_size)

        # Threshold (Simple static threshold for POC)
        threshold = 0.65

        return {
            "filename": file.filename,
            "anomaly_score": round(score, 4),
            "is_anomaly": score > threshold,
            "threshold": threshold,
            "heatmap_base64": heatmap_b64,
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 7. Entry Point & Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Anomaly Detection API")

    # Paths
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Path to dataset root"
    )
    parser.add_argument(
        "--class_name", type=str, default="carpet", help="MVTec class name"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        help="Path to weights",
    )
    parser.add_argument(
        "--memory_bank_path",
        type=str,
        default="./models/memory_bank.pt",
        help="Path to save/load memory bank",
    )

    # Model Params
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for auto-building bank"
    )

    # Server Params
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # Update Global Config
    config.data_root = args.data_root
    config.class_name = args.class_name
    config.weights_path = args.weights_path
    config.memory_bank_path = args.memory_bank_path
    config.img_size = args.img_size
    config.batch_size = args.batch_size
    config.host = args.host
    config.port = args.port

    # Run Uvicorn Programmatically
    # This allows us to use the parsed args inside the app
    print(f"Starting server for class: {config.class_name}")
    uvicorn.run(app, host=config.host, port=config.port)
