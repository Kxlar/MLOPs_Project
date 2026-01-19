import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve

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


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation: Calculate ROC/AUC")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--memory_bank_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)

    # Argparse compatibility
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug_types", nargs="+", default=[])

    return parser.parse_args()


def main():
    args = get_args()
    run(args)


def run(args) -> None:
    """Evaluate image-level and pixel-level ROC/AUC.

    Kept as a separate function so Hydra can call the same logic.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roc_dir = Path(args.output_dir) / args.class_name / "roc"
    roc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    _, test_dataset, _, _ = build_dataloaders(args)

    # 2. Load Model & Memory Bank
    dinov3_model = load_dinov3(args.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)
    memory_bank = torch.load(args.memory_bank_path, map_location=device)

    # --- Image Level Evaluation ---
    print("Running Image-Level Evaluation...")
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
    print(f"Image-Level ROC AUC: {auc_val:.4f}")

    # Plot Histogram
    plt.figure()
    plt.hist(y_score_max[y_true == 0], bins=20, alpha=0.6, label="Good")
    plt.hist(y_score_max[y_true == 1], bins=20, alpha=0.6, label="Defective")
    plt.legend()
    plt.title(f"{args.class_name} - Anomaly Score Histogram")
    plt.savefig(roc_dir / "histogram.png")
    plt.close()

    # --- Pixel Level Evaluation ---
    print("Running Pixel-Level Evaluation...")
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
        print(f"Pixel-Level ROC AUC: {pixel_auc:.4f}")

        fpr, tpr, _ = roc_curve(pixel_y_true, pixel_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {pixel_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{args.class_name} - Pixel ROC")
        plt.legend()
        plt.savefig(roc_dir / "pixel_roc.png")
        plt.close()
    else:
        print("No masks found, skipping pixel evaluation.")


if __name__ == "__main__":
    main()
