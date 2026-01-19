import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock


# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.anomaly_detection.train as train


class TestTrainArguments:
    """Tests for argument parsing logic."""

    def test_get_args_defaults(self):
        """Test argument parsing with minimal required arguments."""
        test_args = [
            "prog",
            "--data_root",
            "./data",
            "--class_name",
            "bottle",
            "--weights_path",
            "./weights/dino.pth",
        ]

        with patch.object(sys, "argv", test_args):
            args = train.get_args()

            assert args.data_root == "./data"
            assert args.class_name == "bottle"
            assert args.weights_path == "./weights/dino.pth"
            assert args.save_path == "./models/memory_bank.pt"  # Default
            assert args.img_size == 224  # Default
            assert args.batch_size == 8  # Default
            assert args.augment is False  # Default

    def test_get_args_missing_required(self):
        """Test that missing required arguments raise SystemExit."""
        test_args = [
            "prog",
            "--data_root",
            "./data",
        ]  # Missing class_name and weights_path

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                train.get_args()


class TestTrainPipeline:
    """Tests for the main execution logic (run function)."""

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Fixture to provide dummy arguments."""
        args = MagicMock()
        args.data_root = "/tmp/data"
        args.class_name = "bottle"
        args.weights_path = "/tmp/weights.pth"
        args.save_path = str(tmp_path / "output" / "memory_bank.pt")
        args.img_size = 224
        args.batch_size = 4
        args.augment = False
        args.aug_types = []
        return args

    @patch("src.anomaly_detection.train.build_dataloaders")
    @patch("src.anomaly_detection.train.load_dinov3")
    @patch("src.anomaly_detection.train.DINOv3FeatureExtractor")
    @patch("src.anomaly_detection.train.build_memory_bank")
    @patch("torch.save")
    def test_run_success(
        self,
        mock_save,
        mock_bank_builder,
        mock_extractor_cls,
        mock_load_dino,
        mock_dataloaders,
        mock_args,
    ):
        """
        Verifies the full pipeline execution:
        1. Dataloaders are built.
        2. Model is loaded and wrapped.
        3. Memory bank is computed.
        4. Result is saved to disk.
        """
        # --- Setup Mocks ---
        # Mock Data
        mock_train_ds = MagicMock()
        mock_train_ds.__len__.return_value = 100
        mock_train_loader = MagicMock()
        # build_dataloaders returns: (train_ds, val_ds, train_loader, val_loader)
        mock_dataloaders.return_value = (mock_train_ds, None, mock_train_loader, None)

        # Mock Model
        mock_dino_model = MagicMock()
        mock_load_dino.return_value = mock_dino_model

        mock_extractor_instance = MagicMock()
        mock_extractor_cls.return_value = mock_extractor_instance
        # Chain calls: extractor.eval().to(device)
        mock_extractor_instance.eval.return_value.to.return_value = (
            mock_extractor_instance
        )

        # Mock Memory Bank
        expected_bank_shape = (100, 384)
        mock_memory_bank_tensor = MagicMock()
        mock_memory_bank_tensor.shape = expected_bank_shape
        mock_bank_builder.return_value = mock_memory_bank_tensor

        # --- Execute ---
        train.run(mock_args)

        # --- Assertions ---
        # 1. Verify Dataloader call
        mock_dataloaders.assert_called_once_with(mock_args)

        # 2. Verify Model Loading
        mock_load_dino.assert_called_once()
        mock_extractor_cls.assert_called_once_with(mock_dino_model)

        # 3. Verify Memory Bank Construction
        mock_bank_builder.assert_called_once()
        # Extract args passed to build_memory_bank to check logic
        call_args = mock_bank_builder.call_args
        assert call_args[0][0] == mock_extractor_instance  # Feature extractor
        assert call_args[0][1] == mock_train_loader  # Loader

        # 4. Verify Saving
        # Check if parent directory creation was attempted
        save_path = Path(mock_args.save_path)
        assert save_path.parent.exists()

        # Check torch.save called with correct tensor and path
        mock_save.assert_called_once_with(mock_memory_bank_tensor, save_path)

    @patch("src.anomaly_detection.train.get_args")
    @patch("src.anomaly_detection.train.run")
    def test_main(self, mock_run, mock_get_args):
        """Test the entry point."""
        mock_get_args.return_value = "dummy_args"
        train.main()
        mock_run.assert_called_once_with("dummy_args")

    @patch("src.anomaly_detection.train.build_dataloaders")
    @patch("src.anomaly_detection.train.load_dinov3")
    @patch("src.anomaly_detection.train.DINOv3FeatureExtractor")
    @patch("src.anomaly_detection.train.build_memory_bank")
    @patch("torch.save")
    def test_run_device_selection(
        self, mock_save, mock_bank, mock_ext, mock_load, mock_data, mock_args
    ):
        """
        Edge case: Verify device logic (CPU vs CUDA).
        Since we cannot force CUDA in a CI environment easily, we check if logic is executed.
        """
        # Setup minimal returns to avoid crashes
        mock_data.return_value = (MagicMock(), None, MagicMock(), None)
        mock_bank.return_value = MagicMock()

        train.run(mock_args)

        # Check that load_dinov3 was called with a torch.device
        args, _ = mock_load.call_args
        assert isinstance(args[1], torch.device)
