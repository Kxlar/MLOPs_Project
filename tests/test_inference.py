import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace
from PIL import Image

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the module to test
import src.anomaly_detection.inference as inference


# --- FIXTURES ---


@pytest.fixture
def mock_args(tmp_path):
    """
    Creates a Namespace object mimicking command line arguments.
    Uses tmp_path to ensure file operations happen in a sandbox.
    """
    return Namespace(
        data_root=str(tmp_path / "data"),
        class_name="bottle",
        weights_path=str(tmp_path / "weights.pt"),
        memory_bank_path=str(tmp_path / "memory_bank.pt"),
        output_dir=str(tmp_path / "results"),
        img_size=224,
        batch_size=1,
        k=5,
        augment=False,
        aug_types=[],
    )


@pytest.fixture
def dummy_image_file(tmp_path):
    """Creates a physical dummy image file for I/O testing."""
    img_path = tmp_path / "test_img.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path


# --- UNIT TESTS: Helper Functions ---


def test_get_args():
    """Test argument parsing defaults and requirements."""
    # Simulate command line args
    test_argv = [
        "inference.py",
        "--data_root",
        "./data",
        "--class_name",
        "metal_nut",
        "--weights_path",
        "model.pt",
        "--memory_bank_path",
        "bank.pt",
    ]

    with patch.object(sys, "argv", test_argv):
        args = inference.get_args()
        assert args.class_name == "metal_nut"
        assert args.k == 10  # Check default
        assert args.img_size == 224


def test_save_heatmap_and_overlay(dummy_image_file, tmp_path):
    """
    Tests the visualization logic.
    Crucial: Ensures matplotlib code runs without errors (even if headless).
    """
    # Inputs
    img_path = str(dummy_image_file)
    am_up = np.random.rand(224, 224).astype(np.float32)
    img_size = 224

    out_heatmap = tmp_path / "heatmap.png"
    out_overlay = tmp_path / "overlay.png"

    # We patch plt to avoid actual window opening, but we let it 'save'
    # effectively by mocking the savefig to create a dummy file or just pass.
    # However, to be robust, let's allow matplotlib to run but use a non-interactive backend
    # or just trust the mock. Here we mock plt to focus on logic flow.

    with patch("src.anomaly_detection.inference.plt") as mock_plt:
        # Run function
        inference.save_heatmap_and_overlay(
            img_path, am_up, img_size, str(out_heatmap), str(out_overlay)
        )

        # Verify Matplotlib calls
        assert mock_plt.figure.call_count == 2
        assert (
            mock_plt.imshow.call_count == 3
        )  # 1 for heatmap, 2 for overlay (bg + alpha)
        assert mock_plt.savefig.call_count == 2
        assert mock_plt.close.call_count == 2


# --- INTEGRATION TESTS: Pipeline ---


@patch("src.anomaly_detection.inference.build_dataloaders")
@patch("src.anomaly_detection.inference.load_dinov3")
@patch("src.anomaly_detection.inference.DINOv3FeatureExtractor")
@patch("src.anomaly_detection.inference.torch.load")
@patch("src.anomaly_detection.inference.compute_anomaly_map")
@patch("src.anomaly_detection.inference.upsample_anomaly_map")
@patch(
    "src.anomaly_detection.inference.save_heatmap_and_overlay"
)  # Mocking the I/O part for the pipeline test
def test_run_inference_pipeline(
    mock_save_viz,
    mock_upsample,
    mock_compute_map,
    mock_torch_load,
    mock_extractor_cls,
    mock_load_dino,
    mock_build_loaders,
    mock_args,
):
    """
    Full pipeline test.
    Verifies that data flows correctly from Loader -> Model -> Map -> Visualization.
    """
    # 1. Setup Mocks

    # Data Loader Mock: returns (None, dataset, None, None)
    # The dataset needs to be iterable and return (img_t, label, path)
    mock_dataset = [
        (torch.randn(3, 224, 224), 0, "/path/to/good/001.png"),  # Good image
        (torch.randn(3, 224, 224), 1, "/path/to/bad/001.png"),  # Defect image
    ]
    mock_build_loaders.return_value = (None, mock_dataset, None, None)

    # Model Mocks
    mock_model = MagicMock()
    mock_load_dino.return_value = mock_model

    mock_extractor_instance = MagicMock()
    mock_extractor_cls.return_value = mock_extractor_instance

    # Memory Bank Mock
    mock_torch_load.return_value = torch.randn(100, 64)  # Dummy bank

    # Anomaly Map Mocks
    mock_compute_map.return_value = torch.rand(14, 14)  # Raw map
    mock_upsample.return_value = np.random.rand(224, 224)  # Upsampled map

    # 2. Run execution
    inference.run(mock_args)

    # 3. Assertions (The "Contract")

    # A. Did we load the model correctly?
    mock_load_dino.assert_called_once_with(mock_args.weights_path, torch.device("cpu"))

    # B. Did we load the memory bank?
    mock_torch_load.assert_called_once()

    # C. Did we process every image in the dataset?
    assert mock_compute_map.call_count == len(mock_dataset)
    assert mock_save_viz.call_count == len(mock_dataset)

    # D. Check naming logic in the loop
    # The second call (index 1) was a defect image
    args, _ = mock_save_viz.call_args_list[1]
    # args: (img_path, am_up, img_size, out_heatmap, out_overlay)
    saved_heatmap_path = str(args[3])
    assert "defect" in saved_heatmap_path

    # The first call (index 0) was a good image
    args_good, _ = mock_save_viz.call_args_list[0]
    saved_heatmap_path_good = str(args_good[3])
    assert "good" in saved_heatmap_path_good


# --- EDGE CASES / SYSTEM CHECKS ---


@patch("src.anomaly_detection.inference.build_dataloaders")
@patch("src.anomaly_detection.inference.load_dinov3")
@patch("src.anomaly_detection.inference.DINOv3FeatureExtractor")
@patch("src.anomaly_detection.inference.torch.load")
def test_directory_creation(mock_load, mock_ext, mock_dino, mock_data, mock_args):
    """Verifies that output directories are automatically created."""
    # Setup minimal mocks to let the script run until directory creation
    mock_data.return_value = (None, [], None, None)  # Empty dataset

    # Run
    inference.run(mock_args)

    # Check
    results_root = Path(mock_args.output_dir) / mock_args.class_name
    assert (results_root / "heatmaps").exists()
    assert (results_root / "overlays").exists()


def test_gpu_handling(mock_args):
    """
    Verifies that the script attempts to use CUDA if available.
    """
    # 1. Force GPU detection to True
    with patch("torch.cuda.is_available", return_value=True):

        # 2. Spy on load_dinov3 to check which device it receives
        with patch("src.anomaly_detection.inference.load_dinov3") as mock_load:

            # 3. FIX: Prevent the data loader from crashing.
            # Make it return empty values so the script can continue.
            with patch(
                "src.anomaly_detection.inference.build_dataloaders",
                return_value=(None, [], None, None),
            ):

                # 4. Force the NEXT step (FeatureExtractor) to fail
                # to stop the script right after load_dinov3 has been called.
                with patch(
                    "src.anomaly_detection.inference.DINOv3FeatureExtractor",
                    side_effect=Exception("Stop Here"),
                ):
                    try:
                        inference.run(mock_args)
                    except Exception:
                        pass

        # Verification
        assert (
            mock_load.called
        ), "load_dinov3 should have been called before the exception"

        args, _ = mock_load.call_args
        # The signature is load_dinov3(weights_path, device) -> args[1] is the device
        assert args[1].type == "cuda"
