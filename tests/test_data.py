from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.anomaly_detection.data import (
    MVTecDataset,
    build_transform,
    build_dataloaders,
    get_defect_type_from_path,
    load_ground_truth_mask,
    save_augmented_dataset,
)

# --- FIXTURES ---


@pytest.fixture
def mock_mvtec_data(tmp_path):
    """
    Creates a temporary directory structure mimicking the MVTec AD dataset.
    Structure:
    /tmp_dir
        /bottle
            /train
                /good (3 images)
            /test
                /good (2 images)
                /broken_large (2 images)
            /ground_truth
                /broken_large (2 masks)
    """
    root = tmp_path / "mvtec_dummy"
    class_name = "bottle"

    # Create directories
    (root / class_name / "train" / "good").mkdir(parents=True)
    (root / class_name / "test" / "good").mkdir(parents=True)
    (root / class_name / "test" / "broken_large").mkdir(parents=True)
    (root / class_name / "ground_truth" / "broken_large").mkdir(parents=True)

    def create_dummy_image(path, size=(100, 100), color=(128, 128, 128)):
        img = Image.new("RGB", size, color)
        img.save(path)

    # Train data
    for i in range(3):
        create_dummy_image(root / class_name / "train" / "good" / f"{i:03d}.png")

    # Test data (Normal)
    for i in range(2):
        create_dummy_image(root / class_name / "test" / "good" / f"{i:03d}.png")

    # Test data (Anomaly) + Masks
    for i in range(2):
        img_name = f"{i:03d}.png"
        mask_name = f"{i:03d}_mask.png"
        create_dummy_image(root / class_name / "test" / "broken_large" / img_name, color=(255, 0, 0))

        # Create a binary mask (L)
        mask = Image.new("L", (100, 100), 255)  # Defect everywhere for testing
        mask.save(root / class_name / "ground_truth" / "broken_large" / mask_name)

    return root, class_name


@pytest.fixture
def mock_args(mock_mvtec_data):
    """Simulates argparse arguments"""
    root, class_name = mock_mvtec_data
    args = Namespace(
        data_root=str(root),
        class_name=class_name,
        img_size=64,  # Small size for faster tests
        batch_size=2,
        augment=False,
        aug_types=[],
        save_aug_path=str(root / "aug_output"),
        aug_multiplier=1,
        rot_degrees=10,
        color_brightness=0.1,
        color_contrast=0.1,
        color_saturation=0.1,
        blur_kernel=3,
    )
    return args


# --- UNIT TESTS (Utility Functions) ---


def test_get_defect_type_from_path():
    # Case 1: Healthy image (good)
    path_good = "/data/mvtec/bottle/test/good/000.png"
    assert get_defect_type_from_path(path_good, "bottle") == "good"

    # Case 2: Specific defect
    path_defect = "/data/mvtec/bottle/test/broken_large/000.png"
    # The script splits on the "test" folder and takes the following element
    assert get_defect_type_from_path(path_defect, "bottle") == "broken_large"

    # Case 3: Weird path (should return unknown or handle the error)
    path_weird = "/data/mvtec/bottle/train/weird/000.png"
    # Note: The current function returns "unknown" if 'good' is not present
    # and 'test' is not present either
    assert get_defect_type_from_path(path_weird, "bottle") == "unknown"


def test_load_ground_truth_mask(mock_mvtec_data):
    root, class_name = mock_mvtec_data
    img_size = 64

    # Case 1: Good image (No mask expected)
    good_img_path = root / class_name / "test" / "good" / "000.png"
    mask = load_ground_truth_mask(good_img_path, root, class_name, img_size)
    assert mask is None

    # Case 2: Anomalous image (Mask expected)
    bad_img_path = root / class_name / "test" / "broken_large" / "000.png"
    mask = load_ground_truth_mask(bad_img_path, root, class_name, img_size)

    assert mask is not None
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (img_size, img_size)
    assert mask.dtype == np.uint8
    # Check binarization (0 or 1)
    assert np.all(np.isin(mask, [0, 1]))


# --- DATASET TESTS (Loading Logic) ---


def test_mvtec_dataset_train_split(mock_mvtec_data):
    root, class_name = mock_mvtec_data
    # Train split should only load "good" folders
    ds = MVTecDataset(root=root, class_name=class_name, split="train")

    assert len(ds) == 3  # We created 3 good train images
    img, label, path = ds[0]

    # Label check (Always 0 for train)
    assert label == 0
    # Type check
    assert isinstance(img, torch.Tensor)


def test_mvtec_dataset_test_split(mock_mvtec_data):
    root, class_name = mock_mvtec_data
    # Test split loads everything
    ds = MVTecDataset(root=root, class_name=class_name, split="test")

    # 2 good + 2 broken = 4
    assert len(ds) == 4

    # Label verification
    labels = ds.labels
    assert labels.count(0) == 2  # 2 good
    assert labels.count(1) == 2  # 2 anomalies


# --- TRANSFORM & AUGMENTATION TESTS ---


def test_build_transform_basic():
    # Standard test (Normalization + Tensor)
    tf = build_transform(img_size=128, is_saving=False)
    dummy_img = Image.new("RGB", (200, 200))
    res = tf(dummy_img)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 128, 128)  # C, H, W


def test_build_transform_saving(mock_args):
    # Saving mode test (No normalization, PIL output)
    mock_args.augment = True
    mock_args.aug_types = ["rotation"]

    tf = build_transform(img_size=128, args=mock_args, is_saving=True)
    dummy_img = Image.new("RGB", (200, 200))
    res = tf(dummy_img)

    assert isinstance(res, Image.Image)  # Must remain a PIL image
    assert res.size == (128, 128)


def test_augmentations_logic(mock_args):
    # Check that adding options correctly modifies the pipeline
    mock_args.augment = True
    mock_args.aug_types = ["rotation", "blur", "color"]

    tf = build_transform(img_size=64, args=mock_args, is_saving=False)
    # It's hard to test visual effects in unit tests,
    # but we can check that the transform runs without errors
    dummy_img = Image.new("RGB", (100, 100))
    res = tf(dummy_img)
    assert res.shape == (3, 64, 64)


# --- PIPELINE / DATALOADER TESTS ---


def test_build_dataloaders(mock_args):
    train_ds, test_ds, train_loader, test_loader = build_dataloaders(mock_args)

    # Batch verification
    batch = next(iter(train_loader))
    imgs, labels, paths = batch

    # Dimensions: (Batch_size, Channel, H, W)
    assert imgs.shape == (
        mock_args.batch_size,
        3,
        mock_args.img_size,
        mock_args.img_size,
    )
    # Labels
    assert labels.shape == (mock_args.batch_size,)
    assert isinstance(paths, tuple) or isinstance(paths, list)
    assert len(paths) == mock_args.batch_size


# --- DATASET GENERATION TEST (Script Functionality) ---


def test_save_augmented_dataset(mock_args, tmp_path):
    # Configuration for saving
    mock_args.augment = True
    mock_args.aug_types = ["rotation"]
    mock_args.aug_multiplier = 2  # 2 augmentations per original image

    # Output directory
    output_root = Path(mock_args.save_aug_path)

    # Execution
    save_augmented_dataset(mock_args)

    # Verification
    expected_output_dir = output_root / f"{mock_args.class_name}_augmented" / "train" / "good"
    assert expected_output_dir.exists()

    files = list(expected_output_dir.glob("*.png"))
    assert len(files) == 6
