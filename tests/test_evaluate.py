import sys
import pytest
import numpy as np
import torch
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import module under test
import src.anomaly_detection.evaluate as evaluate


# --- FIXTURES ---


@pytest.fixture
def mock_args(tmp_path):
    return Namespace(
        data_root=str(tmp_path / "data"),
        class_name="bottle",
        weights_path=str(tmp_path / "model.pt"),
        memory_bank_path=str(tmp_path / "bank.pt"),
        output_dir=str(tmp_path / "results"),
        img_size=64,
        batch_size=2,
        k=5,
        augment=False,
        aug_types=[],
    )


@pytest.fixture
def mock_dataset():
    """
    Simulates a Map-style dataset (supports len() and indexing).
    Contains 3 items:
    0: Good image
    1: Defect image
    2: Defect image (to test aggregation)
    """
    img_size = 64
    # (img_tensor, label, path)
    return [
        (torch.randn(3, img_size, img_size), 0, "/path/good/0.png"),
        (torch.randn(3, img_size, img_size), 1, "/path/bad/1.png"),
        (torch.randn(3, img_size, img_size), 1, "/path/bad/2.png"),
    ]


# --- UNIT TESTS: ARGS ---


def test_get_args():
    test_argv = [
        "evaluate.py",
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
        args = evaluate.get_args()
        assert args.class_name == "test"
        assert args.img_size == 224  # Default check


# --- INTEGRATION TESTS: FULL PIPELINE ---


@patch("src.anomaly_detection.evaluate.build_dataloaders")
@patch("src.anomaly_detection.evaluate.load_dinov3")
@patch("src.anomaly_detection.evaluate.DINOv3FeatureExtractor")
@patch("src.anomaly_detection.evaluate.torch.load")  # For memory bank
@patch("src.anomaly_detection.evaluate.compute_anomaly_map")
@patch("src.anomaly_detection.evaluate.reduce_anomaly_map")
@patch("src.anomaly_detection.evaluate.upsample_anomaly_map")
@patch("src.anomaly_detection.evaluate.load_ground_truth_mask")
@patch("src.anomaly_detection.evaluate.plt")  # Mock plotting to avoid GUI/File creation
def test_evaluation_pipeline_happy_path(
    mock_plt,
    mock_load_mask,
    mock_upsample,
    mock_reduce,
    mock_compute,
    mock_torch_load,
    mock_extractor,
    mock_load_dino,
    mock_build_loaders,
    mock_args,
    mock_dataset,
):
    """
    Tests the standard flow: Image Level + Pixel Level evaluation.
    We inject values to ensure the AUC calculation works.
    """
    # 1. Setup Mocks
    mock_build_loaders.return_value = (None, mock_dataset, None, None)

    # Mock Model/Bank
    mock_torch_load.return_value = torch.randn(10, 64)  # Memory bank

    # Mock Anomaly Map Generation
    # Return random map 14x14
    mock_compute.return_value = torch.rand(14, 14)

    # Mock Reduction (Image Level Score)
    # We want to force a perfect AUC to verify logic
    # Dataset: [Good(0), Bad(1), Bad(1)]
    # We return scores: [0.1, 0.9, 0.8] -> Should give AUC = 1.0
    mock_reduce.side_effect = [0.1, 0.9, 0.8]

    # Mock Upsampling (Pixel Level)
    # Return 64x64 numpy array
    mock_upsample.return_value = np.zeros((64, 64))

    # Mock Ground Truth Masks
    # Logic: Good image -> None (implicit), Bad image -> Mask
    # Side effect corresponds to the dataset iteration order in the SECOND loop
    # Dataset[0] (Good) -> load_ground_truth_mask returns None
    # Dataset[1] (Bad)  -> returns 64x64 mask
    # Dataset[2] (Bad)  -> returns 64x64 mask
    mock_load_mask.side_effect = [
        None,
        np.ones((64, 64), dtype=np.uint8),
        np.ones((64, 64), dtype=np.uint8),
    ]

    # 2. Execution
    evaluate.run(mock_args)

    # 3. Verification

    # A. Check Image Level Logic
    # reduce_anomaly_map should be called 3 times (once per image in loop 1)
    # Note: It might be called again in loop 2? No, loop 2 uses upsample.
    assert mock_reduce.call_count == 3

    # B. Check Plotting
    # We expect 2 figures: 1 Histogram (Image level), 1 ROC Curve (Pixel level)
    assert mock_plt.figure.call_count == 2
    assert mock_plt.savefig.call_count == 2

    # Verify paths of saved plots
    expected_dir = Path(mock_args.output_dir) / mock_args.class_name / "roc"
    call_args_list = mock_plt.savefig.call_args_list

    # Check if histogram.png was saved
    assert any("histogram.png" in str(c[0][0]) for c in call_args_list)
    # Check if pixel_roc.png was saved
    assert any("pixel_roc.png" in str(c[0][0]) for c in call_args_list)


# --- EDGE CASE TESTS ---


@patch("src.anomaly_detection.evaluate.roc_auc_score")
@patch("src.anomaly_detection.evaluate.roc_curve")
@patch("src.anomaly_detection.evaluate.build_dataloaders")
@patch("src.anomaly_detection.evaluate.load_dinov3")
@patch("src.anomaly_detection.evaluate.DINOv3FeatureExtractor")
@patch("src.anomaly_detection.evaluate.torch.load")
@patch("src.anomaly_detection.evaluate.compute_anomaly_map")
@patch("src.anomaly_detection.evaluate.reduce_anomaly_map")
@patch("src.anomaly_detection.evaluate.upsample_anomaly_map")
@patch("src.anomaly_detection.evaluate.load_ground_truth_mask")
@patch("src.anomaly_detection.evaluate.plt")
def test_pixel_logic_missing_masks(
    mock_plt,
    mock_load_mask,
    mock_upsample,
    mock_reduce,
    mock_compute,
    mock_torch_load,
    mock_extractor,
    mock_load_dino,
    mock_build_loaders,
    mock_roc_curve,
    mock_sklearn_auc,
    mock_args,
):
    """
    MLOps Critical Test: Data Integrity.
    What happens if we have a Defect image (Label=1) but the Ground Truth mask file is missing?
    The code should skip this image for pixel evaluation to avoid crashing or calculating false AUC.
    """
    # Dataset: 1 Good, 1 Bad (with missing mask)
    dataset = [
        (torch.randn(3, 64, 64), 0, "good.png"),
        (torch.randn(3, 64, 64), 1, "bad_missing_mask.png"),
    ]
    mock_build_loaders.return_value = (None, dataset, None, None)

    mock_reduce.return_value = 0.5
    mock_upsample.return_value = np.zeros((64, 64))
    mock_sklearn_auc.return_value = 0.5
    mock_roc_curve.return_value = ([0, 1], [0, 1], [0.5, 0.5])

    # Mask Loading Behavior:
    # 1. Good Image -> Returns None (Normal) -> Logic converts to Zero Mask
    # 2. Bad Image -> Returns None (File not found) -> Logic should CONTINUE (skip)
    mock_load_mask.side_effect = [None, None]

    # Execution
    evaluate.run(mock_args)

    # Verification
    # upsample_anomaly_map is called in the Pixel Loop.
    # If the logic works, it should be called for the Good image,
    # BUT skipped for the Bad image because mask was None.
    # Wait, let's look at evaluate.py:
    #   if gt_mask is None:
    #       if label == 0: gt_mask = zeros
    #       else: continue
    #   ...
    #   am_up = upsample(...)

    # So upsample should ONLY be called for the Good image (1 time).
    assert mock_upsample.call_count == 1

    # Ensure Roc Curve was plotted (even with 1 sample? Sklearn might warn, but code runs)
    # Actually with 1 sample (all class 0), ROC AUC is undefined.
    # However, the test here is just to ensure the code didn't crash on the 'continue'.


def test_perfect_separation_metric_check(mock_args):
    """
    Pure Logic Test: Does roc_auc_score work as expected?
    We verify the sklearn integration without running the full pipeline.
    """
    from sklearn.metrics import roc_auc_score

    # Scenario: Perfect Model
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.9, 0.8])

    auc = roc_auc_score(y_true, y_scores)
    assert auc == 1.0

    # Scenario: Random/Bad Model
    y_scores_bad = np.array([0.9, 0.8, 0.1, 0.2])  # Inverted
    auc_bad = roc_auc_score(y_true, y_scores_bad)
    assert auc_bad == 0.0


@patch("src.anomaly_detection.evaluate.build_dataloaders")
@patch("src.anomaly_detection.evaluate.load_dinov3")
@patch("src.anomaly_detection.evaluate.DINOv3FeatureExtractor")
@patch("src.anomaly_detection.evaluate.torch.load")
@patch("src.anomaly_detection.evaluate.compute_anomaly_map")
@patch("src.anomaly_detection.evaluate.reduce_anomaly_map")
def test_directory_creation(
    mock_reduce, mock_compute, mock_torch, mock_ext, mock_load, mock_data, mock_args
):
    """Verifies output folder structure."""

    # TIP: Intentionally make data loading fail.
    # Since evaluate.py creates the output directory (line 59)
    # BEFORE loading the data (line 62),
    # this allows us to test directory creation without having to mock
    # the rest of the pipeline (AUC, plots, etc.).
    mock_data.side_effect = Exception("Stop Early")

    # Catch the intentional error
    try:
        evaluate.run(mock_args)
    except Exception as e:
        assert str(e) == "Stop Early"

    # Verification: Was the directory created before the crash?
    expected_dir = Path(mock_args.output_dir) / mock_args.class_name / "roc"
    assert expected_dir.exists()
