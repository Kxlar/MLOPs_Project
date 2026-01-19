import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.anomaly_detection.model as model_module
from src.anomaly_detection.model import (
    DINOv3FeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
    reduce_anomaly_map,
    upsample_anomaly_map,
)

# --- FIXTURES AND MOCKS ---


class MockViT(nn.Module):
    """
    Simulates the specific behavior of the vit_base model used in DinoV3.
    It returns a dictionary with 'x_norm_patchtokens'.
    """

    def __init__(self, embed_dim=64, n_patches=196):  # 14x14 = 196
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches

    def forward_features(self, x):
        # x shape: [B, 3, H, W]
        B = x.shape[0]
        # Simulate output: [B, N_patches, Embed_Dim]
        tokens = torch.randn(B, self.n_patches, self.embed_dim)
        return {"x_norm_patchtokens": tokens}


@pytest.fixture
def mock_dino_model():
    return MockViT(embed_dim=64, n_patches=196)  # 14x14 patches


@pytest.fixture
def feature_extractor(mock_dino_model):
    return DINOv3FeatureExtractor(mock_dino_model)


@pytest.fixture
def mock_dataloader():
    """Generates dummy batches for build_memory_bank testing"""
    # 2 batches, each with 2 images of size 224x224
    batch_1 = (torch.randn(2, 3, 224, 224), torch.zeros(2), ["p1", "p2"])
    batch_2 = (torch.randn(2, 3, 224, 224), torch.zeros(2), ["p3", "p4"])
    return [batch_1, batch_2]


# --- TESTS: MODEL LOADING ---


@patch.object(model_module, "vit_base")
@patch("torch.load")
def test_load_dinov3_structure(mock_torch_load, mock_vit_base):
    """
    Tests if the model loads state dicts correctly and handles specific keys
    (teacher/model) and filters out technical buffers (bias_mask).
    """
    # Setup Mocks
    mock_model_instance = MagicMock()
    mock_vit_base.return_value = mock_model_instance

    # Simulation 1: Checkpoint with 'teacher' key and unwanted 'bias_mask'
    mock_torch_load.return_value = {
        "teacher": {"weight": 1, "bias_mask_to_ignore": 0}  # Should be filtered
    }

    device = torch.device("cpu")
    # On appelle la fonction via le module importé pour être cohérent
    model = model_module.load_dinov3("fake_path.pth", device)

    # Assertions
    mock_vit_base.assert_called_once()
    # Check if load_state_dict was called with filtered dict (no bias_mask)
    args, _ = mock_model_instance.load_state_dict.call_args
    loaded_dict = args[0]
    assert "bias_mask_to_ignore" not in loaded_dict
    assert "weight" in loaded_dict
    assert model == mock_model_instance


@patch.object(model_module, "vit_base")
@patch("torch.load")
def test_load_dinov3_fallback_key(mock_torch_load, mock_vit_base):
    """Tests fallback to 'model' key if 'teacher' is missing."""
    mock_model_instance = MagicMock()
    mock_vit_base.return_value = mock_model_instance

    # Simulation 2: Checkpoint with 'model' key
    mock_torch_load.return_value = {"model": {"weight": 2}}

    model_module.load_dinov3("fake_path.pth", torch.device("cpu"))

    args, _ = mock_model_instance.load_state_dict.call_args
    assert args[0]["weight"] == 2


# --- TESTS: FEATURE EXTRACTOR ---


def test_feature_extractor_shapes(feature_extractor):
    """
    Verifies that the extractor reshapes [B, N, D] -> [B, D, H, W] correctly.
    """
    # Input: Batch=2, Channels=3, Size=224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = feature_extractor(dummy_input)

    # Expected: Batch=2, Dim=64, H=14, W=14 (since sqrt(196)=14)
    assert output.shape == (2, 64, 14, 14)
    assert output.dtype == torch.float32


def test_feature_extractor_invalid_patches():
    """
    Edge Case: What if the model returns a number of patches that isn't a perfect square?
    """
    # Create model producing 200 patches (sqrt(200) is not int)
    bad_model = MockViT(n_patches=200)
    extractor = DINOv3FeatureExtractor(bad_model)

    dummy_input = torch.randn(1, 3, 224, 224)

    with pytest.raises(AssertionError) as excinfo:
        extractor(dummy_input)
    assert "Patch count 200 is not a perfect square" in str(excinfo.value)


# --- TESTS: MEMORY BANK (CORE LOGIC) ---


def test_build_memory_bank(feature_extractor, mock_dataloader):
    """
    Integration test: Loops through loader, extracts, normalizes, and concatenates.
    """
    device = torch.device("cpu")

    # Run function
    memory_bank = build_memory_bank(feature_extractor, mock_dataloader, device)

    # Calculations for expected shape:
    # 2 batches * 2 images/batch = 4 images
    # 4 images * 196 patches/image = 784 total vectors
    # Embedding dim = 64
    assert memory_bank.shape == (784, 64)

    # Critical Production Check: Normalization
    # Features must be L2 normalized for Cosine/Euclidean distance to work robustly
    norms = torch.norm(memory_bank, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# --- TESTS: ANOMALY MAP COMPUTATION ---


def test_compute_anomaly_map(feature_extractor):
    """
    Verifies the Nearest Neighbor logic and output map generation.
    """
    # Setup
    img_t = torch.randn(3, 224, 224)  # Single image

    # Create a synthetic memory bank [100 vectors, 64 dim]
    memory_bank = torch.randn(100, 64)
    memory_bank = torch.nn.functional.normalize(memory_bank, dim=1)

    # Run
    k = 5
    anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=k)

    # Checks
    # Output should be 2D spatial map [14, 14]
    assert anomaly_map.shape == (14, 14)
    # Should be on CPU
    assert anomaly_map.device == torch.device("cpu")
    # Values should be positive (distances)
    assert (anomaly_map >= 0).all()


# --- TESTS: POST PROCESSING ---


def test_reduce_anomaly_map():
    # Setup dummy map 2x2
    am = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Case Max
    assert reduce_anomaly_map(am, "max") == 4.0

    # Case Mean
    assert reduce_anomaly_map(am, "mean") == 2.5

    # Case Invalid
    with pytest.raises(ValueError):
        reduce_anomaly_map(am, "median")


def test_upsample_anomaly_map():
    """
    Verifies interpolation from Patch Size -> Image Size.
    """
    # Small map 14x14
    am = torch.rand(14, 14)
    target_size = 224

    # Run
    result = upsample_anomaly_map(am, target_size)

    # Check type (must be numpy for further processing/saving)
    assert isinstance(result, np.ndarray)

    # Check shape
    assert result.shape == (target_size, target_size)

    # Check values range is preserved roughly (interpolation shouldn't explode values)
    assert result.max() <= 1.0 and result.min() >= 0.0
