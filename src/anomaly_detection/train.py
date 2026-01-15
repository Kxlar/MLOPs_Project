import sys
import argparse
from pathlib import Path
import torch

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.data import build_dataloaders
from src.anomaly_detection.model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    build_memory_bank,
)


def get_args():
    parser = argparse.ArgumentParser(description="Train: Build Memory Bank")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument(
        "--weights_path", type=str, required=True, help="Path to DinoV3 weights"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./models/memory_bank.pt",
        help="Where to save the memory bank tensor",
    )

    # Data params matching data.py
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--augment", action="store_true"
    )  # Usually False for building memory bank
    parser.add_argument("--aug_types", nargs="+", default=[])

    return parser.parse_args()


def main():
    args = get_args()
    run(args)


def run(args) -> None:
    """Build and save the memory bank.

    Kept as a separate function so Hydra can call the same logic.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    # For memory bank construction, we typically use standard train set (no huge augmentation needed mostly)
    train_ds, _, train_loader, _ = build_dataloaders(args)
    print(f"Training images: {len(train_ds)}")

    # 2. Load Model
    dinov3_model = load_dinov3(args.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)

    # 3. Build Memory Bank
    print("Building memory bank...")
    memory_bank = build_memory_bank(feature_extractor, train_loader, device)
    print(f"Memory Bank shape: {memory_bank.shape}")

    # 4. Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory_bank, save_path)
    print(f"Memory bank saved to {save_path}")


if __name__ == "__main__":
    main()
