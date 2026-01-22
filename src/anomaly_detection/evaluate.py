import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from loguru import logger

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.data import build_dataloaders, load_ground_truth_mask
from src.anomaly_detection.model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    compute_anomaly_map,
    reduce_anomaly_map,
    upsample_anomaly_map,
)


def setup_logger() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        "logs/evaluate_{time}.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
    )


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation: Calculate ROC/AUC")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--memory_bank_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument(
        "--scores_jsonl",
        type=str,
        default=None,
        help="If provided, load anomaly scores from JSONL and only generate histogram (and AUC if possible)",
    )
    parser.add_argument("--hist_only", action="store_true", help="Skip pixel-level evaluation")

    # Argparse compatibility
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug_types", nargs="+", default=[])

    return parser.parse_args()


def main():
    setup_logger()
    args = get_args()

    log = logger.bind(
        class_name=args.class_name,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        k=args.k,
        scores_jsonl=args.scores_jsonl,
        hist_only=args.hist_only,
    )

    log.info("Starting evaluation")

    try:
        run(args, log)
        log.success("Evaluation completed")
    except Exception:
        log.exception("Evaluation failed")
        raise


def run(args, log=logger) -> None:
    """Evaluate image-level and pixel-level ROC/AUC.

    Kept as a separate function so Hydra can call the same logic.
    """

    roc_dir = Path(args.output_dir) / args.class_name / "roc"
    roc_dir.mkdir(parents=True, exist_ok=True)

    if args.scores_jsonl is not None:
        import json

        y_true = []
        y_score_max = []
        with open(args.scores_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("anomaly_score") is None or rec.get("label") is None:
                    continue
                y_true.append(int(rec["label"]))
                y_score_max.append(float(rec["anomaly_score"]))

        y_true = np.asarray(y_true)
        y_score_max = np.asarray(y_score_max)
        if len(y_true) == 0:
            raise ValueError(f"No usable records found in scores file: {args.scores_jsonl}")

        # Compute AUC if both classes are present
        if len(np.unique(y_true)) > 1:
            auc_val = roc_auc_score(y_true, y_score_max)
            log.info("Image-Level ROC AUC (from scores): {:.4f}", auc_val)
        else:
            log.warning("Only one class present in scores; skipping AUC.")

        # Plot Histogram
        plt.figure()
        plt.hist(y_score_max[y_true == 0], bins=20, alpha=0.6, label="Good")
        plt.hist(y_score_max[y_true == 1], bins=20, alpha=0.6, label="Defective")
        plt.legend()
        plt.title(f"{args.class_name} - Anomaly Score Histogram")
        plt.savefig(roc_dir / "histogram.png")
        plt.close()

        log.info("Saved histogram to {}", roc_dir / "histogram.png")
        return

    if args.data_root is None or args.weights_path is None or args.memory_bank_path is None:
        raise ValueError(
            "When --scores_jsonl is not provided, --data_root/--weights_path/--memory_bank_path are required"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    _, test_dataset, _, _ = build_dataloaders(args)

    # 2. Load Model & Memory Bank
    dinov3_model = load_dinov3(args.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)
    memory_bank = torch.load(args.memory_bank_path, map_location=device)

    # --- Image Level Evaluation ---
    log.info("Running Image-Level Evaluation...")
    y_true = []
    y_score_max = []

    for i in range(len(test_dataset)):
        img_t, label, _ = test_dataset[i]
        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=args.k)

        y_true.append(label)
        y_score_max.append(reduce_anomaly_map(anomaly_map, mode="max"))

    y_true = np.array(y_true)
    y_score_max = np.array(y_score_max)

    auc_val = roc_auc_score(y_true, y_score_max)
    log.info("Image-Level ROC AUC: {:.4f}", auc_val)

    # Plot Histogram
    plt.figure()
    plt.hist(y_score_max[y_true == 0], bins=20, alpha=0.6, label="Good")
    plt.hist(y_score_max[y_true == 1], bins=20, alpha=0.6, label="Defective")
    plt.legend()
    plt.title(f"{args.class_name} - Anomaly Score Histogram")
    plt.savefig(roc_dir / "histogram.png")
    plt.close()

    # --- Pixel Level Evaluation ---
    log.info("Running Pixel-Level Evaluation...")
    if args.hist_only:
        log.info("hist_only enabled; skipping pixel evaluation.")
        return

    pixel_y_true = []
    pixel_scores = []

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]

        gt_mask = load_ground_truth_mask(path, args.data_root, args.class_name, args.img_size)
        if gt_mask is None:
            if label == 0:
                gt_mask = np.zeros((args.img_size, args.img_size), dtype=np.uint8)
            else:
                continue

        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=args.k)
        am_up = upsample_anomaly_map(anomaly_map, args.img_size)

        pixel_y_true.append(gt_mask.flatten())
        pixel_scores.append(am_up.flatten())

    if pixel_y_true:
        pixel_y_true = np.concatenate(pixel_y_true)
        pixel_scores = np.concatenate(pixel_scores)

        pixel_auc = roc_auc_score(pixel_y_true, pixel_scores)
        log.info("Pixel-Level ROC AUC: {:.4f}", pixel_auc)

        fpr, tpr, _ = roc_curve(pixel_y_true, pixel_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {pixel_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{args.class_name} - Pixel ROC")
        plt.legend()
        plt.savefig(roc_dir / "pixel_roc.png")
        plt.close()
    else:
        log.warning("No masks found, skipping pixel evaluation.")


if __name__ == "__main__":
    main()
