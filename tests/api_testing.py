import io
import sys
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from MLOPs_Project.src.anomaly_detection.API.api import app, ml_models

# --- Fixtures ---


@pytest.fixture
def mock_dependencies():
    """
    Patches heavy dependencies (model loading, inference) to ensure fast tests.
    """
    with (
        patch("src.anomaly_detection.api.load_dinov3") as mock_load,
        patch("src.anomaly_detection.api.torch.load") as mock_torch_load,
        patch("src.anomaly_detection.api.run_auto_build") as mock_build,
        patch("src.anomaly_detection.api.compute_anomaly_map") as mock_compute,
    ):

        # Mock model objects
        mock_load.return_value = MagicMock()
        mock_torch_load.return_value = MagicMock()

        # Return a dummy tensor for the anomaly map (batch=1, channels=1, h=56, w=56)
        mock_compute.return_value = torch.rand((1, 1, 56, 56))

        yield


@pytest.fixture
def client(mock_dependencies):
    """
    Injects mock models into the global API state and provides a TestClient.
    """
    # Manually populate the global state to pass API availability checks
    ml_models["feature_extractor"] = MagicMock()
    ml_models["memory_bank"] = MagicMock()

    with TestClient(app) as c:
        yield c

    # Clean up global state after tests to avoid side effects
    ml_models.clear()


def create_dummy_image():
    """Creates a small valid PNG image in memory."""
    data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# --- Tests ---


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] is True


def test_prediction_flow(client):
    img_bytes = create_dummy_image()
    files = {"file": ("test_image.png", img_bytes, "image/png")}

    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert "heatmap_base64" in data
    assert isinstance(data["anomaly_score"], float)
