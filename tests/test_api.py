import io
from fastapi.testclient import TestClient
from PIL import Image
from pathlib import Path
import numpy as np
import pytest
import torch

import sys

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.api import app


def create_dummy_image():
    """Creates a small valid PNG image in memory for testing."""
    # Create a random 224x224 image
    data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(data)

    # Save to bytes buffer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_check():
    """Test the /health endpoint to ensure models are loaded."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["models_loaded"] is True


def test_prediction_flow():
    """Test the /predict endpoint with a dummy image."""
    # Create a dummy image file
    img_bytes = create_dummy_image()

    with TestClient(app) as client:
        # Simulate file upload
        files = {"file": ("test_image.png", img_bytes, "image/png")}

        response = client.post("/predict", files=files)

        # 1. Assert Status Code
        assert response.status_code == 200

        # 2. Assert Response Structure
        data = response.json()
        assert "anomaly_score" in data
        assert "is_anomaly" in data
        assert "heatmap_base64" in data
        assert isinstance(data["anomaly_score"], float)
