import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace
from PIL import Image

from src.anomaly_detection import inference

# --- FIXTURES ---


@pytest.fixture
def mock_args(tmp_path):
    return Namespace(
        data_root=str(tmp_path / "data"),
        class_name="bottle",
        weights_path=str(tmp_path / "weights.pt"),
        memory_bank_path=str(tmp_path / "memory_bank.pt"),
        output_dir=str(tmp_path / "results"),
        output_name=None,
        img_size=224,
        batch_size=1,
        k=5,
        augment=False,
        aug_types=[],
        split="test",
        save_heatmaps=True,
        save_overlays=True,
        heatmaps_only=False,
        scores_jsonl=None,
    )


@pytest.fixture
def dummy_image_file(tmp_path):
    img_path = tmp_path / "test_img.png"
    Image.new("RGB", (100, 100), color="red").save(img_path)
    return img_path


# --- TESTS ---


def test_get_args():
    test_argv = [
        "inference.py",
        "--data_root",
        ".",
        "--class_name",
        "test",
        "--weights_path",
        "w",
        "--memory_bank_path",
        "m",
    ]
    with patch.object(sys, "argv", test_argv):
        args = inference.get_args()
        assert args.class_name == "test"


def test_save_heatmap_and_overlay(dummy_image_file, tmp_path):
    img_path = str(dummy_image_file)
    am_up = np.random.rand(224, 224).astype(np.float32)
    # Using string patch here too for consistency, though object patch might work for plt
    with patch("src.anomaly_detection.inference.plt") as mock_plt:
        inference.save_heatmap_and_overlay(
            img_path, am_up, 224, str(tmp_path / "h.png"), str(tmp_path / "o.png")
        )
        assert mock_plt.savefig.call_count == 2


@patch.object(inference, "MVTecDataset")
@patch.object(inference, "build_transform")
@patch.object(inference, "load_dinov3")
@patch.object(inference, "DINOv3FeatureExtractor")
@patch.object(inference, "torch")
@patch.object(inference, "compute_anomaly_map")
@patch.object(inference, "upsample_anomaly_map")
@patch.object(inference, "save_heatmap_and_overlay")
def test_run_inference_pipeline(
    mock_save_viz,
    mock_upsample,
    mock_compute_map,
    mock_torch,
    mock_extractor_cls,
    mock_load_dino,
    mock_transform,
    mock_dataset_cls,
    mock_args,
):
    # Mock Data
    mock_dataset = [
        (torch.randn(3, 224, 224), 0, "/path/good/1.png"),
        (torch.randn(3, 224, 224), 1, "/path/bad/1.png"),
    ]
    mock_instance = MagicMock()
    mock_instance.__len__.return_value = len(mock_dataset)
    mock_instance.__getitem__.side_effect = mock_dataset
    mock_dataset_cls.return_value = mock_instance

    # Mock Model/Bank
    mock_torch.load.return_value = torch.randn(10, 64)
    mock_compute_map.return_value = torch.rand(14, 14)
    mock_upsample.return_value = np.random.rand(224, 224)

    inference.run(mock_args)

    assert mock_compute_map.call_count == 2
    assert mock_save_viz.call_count == 2

    args_bad, _ = mock_save_viz.call_args_list[1]
    assert "defect" in str(args_bad[3])


@patch.object(inference, "MVTecDataset")
def test_directory_creation(mock_dataset_cls, mock_args):
    mock_dataset_cls.side_effect = Exception("Stop Early")
    try:
        inference.run(mock_args)
    except Exception:
        pass

    results_root = Path(mock_args.output_dir) / mock_args.class_name
    assert (results_root / "heatmaps").exists()


def test_gpu_handling(mock_args):
    """
    Verifies that the script attempts to use CUDA if available.
    """
    # 1. Force GPU detection to True
    with patch("torch.cuda.is_available", return_value=True):

        # 2. Spy on load_dinov3 to check which device it receives
        with patch.object(inference, "load_dinov3") as mock_load:

            # 3. Mock Dataset (return empty to skip loop)
            with patch.object(inference, "MVTecDataset") as mock_ds:
                mock_ds.return_value.__len__.return_value = 0

                # 4. Mock build_transform (called before dataset)
                with patch.object(inference, "build_transform"):

                    # 5. NEW: Mock torch to handle memory_bank loading
                    with patch.object(inference, "torch") as mock_torch:
                        # Setup fake memory bank return
                        mock_torch.load.return_value = MagicMock()
                        # Ensure checks like torch.cuda.is_available still work if called via this mock
                        mock_torch.cuda.is_available.return_value = True
                        mock_torch.device = torch.device

                        # Run
                        inference.run(mock_args)

        # Verification
        assert mock_load.called
        args, _ = mock_load.call_args
        assert args[1].type == "cuda"
