import sys
import argparse
from pathlib import Path
import torch

from loguru import logger


def setup_logger() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        "logs/train_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )


# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.data import build_dataloaders
from src.anomaly_detection.model import load_dinov3, DINOv3FeatureExtractor, build_memory_bank


def get_args():
    parser = argparse.ArgumentParser(description="Train: Build Memory Bank")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True, help="Path to DinoV3 weights")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./models/memory_bank.pt",
        help="Where to save the memory bank tensor",
    )

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug_types", nargs="+", default=[])

    return parser.parse_args()


def main():
    setup_logger()
    args = get_args()

    log = logger.bind(
        class_name=args.class_name,
        data_root=args.data_root,
        weights_path=args.weights_path,
        save_path=args.save_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=args.augment,
    )

    log.info("Starting training (memory bank build)")

    try:
        run(args, log)
        log.success("Training completed")
    except Exception:
        log.exception("Training failed")
        raise


def run(args, log=logger) -> None:
    """Build and save the memory bank."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: {}", device)

    # 1) Load Data
    train_ds, _, train_loader, _ = build_dataloaders(args)
    log.info("Training images: {}", len(train_ds))

    # 2) Load Model
    log.info("Loading DINOv3 model...")
    dinov3_model = load_dinov3(args.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)

    # 3) Build Memory Bank
    log.info("Building memory bank...")
    memory_bank = build_memory_bank(feature_extractor, train_loader, device)
    log.info("Memory Bank shape: {}", tuple(memory_bank.shape))

    # 4) Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory_bank, save_path)
    log.info("Memory bank saved to {}", save_path)


if __name__ == "__main__":
    main()
