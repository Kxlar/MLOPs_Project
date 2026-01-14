from __future__ import annotations

import time
from pathlib import Path

import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc

from src.anomaly_detection.data import build_dataloaders, load_ground_truth_mask
from src.anomaly_detection.model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
    reduce_anomaly_map,
    upsample_anomaly_map,
)

DEFAULT_WEIGHTS = "./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DEFAULT_DATA_ROOT = "./data"
DEFAULT_CLASS_NAME = "carpet"
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 8
DEFAULT_K = 10


def save_heatmap_and_overlay(
    img_path: str,
    am_up: np.ndarray,
    img_size: int,
    out_heatmap: Path,
    out_overlay: Path,
) -> None:
    """Save standalone heatmap and overlay visualizations for one image."""
    am_norm = (am_up - am_up.min()) / (am_up.max() - am_up.min() + 1e-8)

    orig_img = Image.open(img_path).convert("RGB")
    orig_img = orig_img.resize((img_size, img_size))
    orig_np = np.array(orig_img)

    # heatmap
    plt.figure()
    plt.axis("off")
    plt.imshow(am_norm, cmap="jet")
    plt.tight_layout(pad=0)
    plt.savefig(out_heatmap, bbox_inches="tight", pad_inches=0)
    plt.close()

    # overlay
    plt.figure()
    plt.axis("off")
    plt.imshow(orig_np)
    plt.imshow(am_norm, cmap="jet", alpha=0.5)
    plt.tight_layout(pad=0)
    plt.savefig(out_overlay, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_baseline(
    weights: str | Path = DEFAULT_WEIGHTS,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    class_name: str = DEFAULT_CLASS_NAME,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    k: int = DEFAULT_K,
    show_plots: bool = False,
) -> None:
    """Run the kNN baseline and persist figures/heatmaps."""

    results_root = Path("./reports/figures") / class_name
    heatmap_dir = results_root / "heatmaps"
    overlay_dir = results_root / "overlays"
    roc_dir = results_root / "roc"

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)

    print("DATA_ROOT:", data_root)
    print("CLASS_NAME:", class_name)
    print("DINOv3_WEIGHTS:", weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset, test_dataset, train_loader = build_dataloaders(
        data_root, class_name, img_size, batch_size=batch_size
    )
    print("Train images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    test_labels = np.array(test_dataset.labels)
    n_good = int(np.sum(test_labels == 0))
    n_def = int(np.sum(test_labels == 1))
    print(
        f"Test set stats for '{class_name}': good={n_good}, defective={n_def}, total={len(test_dataset)}"
    )

    dinov3_model = load_dinov3(weights, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)

    memory_bank = build_memory_bank(feature_extractor, train_loader, device)
    print("Memory bank shape:", tuple(memory_bank.shape))

    y_true: list[int] = []
    y_score_max: list[float] = []
    y_score_mean: list[float] = []
    inference_times: list[float] = []

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]

        start = time.time()
        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=k)
        end = time.time()

        inference_times.append(end - start)

        y_true.append(label)
        y_score_max.append(reduce_anomaly_map(anomaly_map, mode="max"))
        y_score_mean.append(reduce_anomaly_map(anomaly_map, mode="mean"))

    y_true_np = np.array(y_true)
    y_score_max_np = np.array(y_score_max)
    y_score_mean_np = np.array(y_score_mean)

    auc_max = roc_auc_score(y_true_np, y_score_max_np)
    auc_mean = roc_auc_score(y_true_np, y_score_mean_np)

    avg_ms = float(np.mean(inference_times)) * 1000
    std_ms = float(np.std(inference_times)) * 1000

    print(f"[{class_name}] ROC AUC (max over patches):  {auc_max:.4f}")
    print(f"[{class_name}] ROC AUC (mean over patches): {auc_mean:.4f}")
    print(
        f"[{class_name}] Avg inference time: {avg_ms:.2f} ms (std {std_ms:.2f} ms, N={len(inference_times)})"
    )

    plt.hist(y_score_max_np[y_true_np == 0], bins=20, alpha=0.6, label="good")
    plt.hist(y_score_max_np[y_true_np == 1], bins=20, alpha=0.6, label="defective")
    plt.legend()
    plt.title(f"{class_name} - Baseline image anomaly scores (max)")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.savefig(roc_dir / "image_score_hist_max.png", bbox_inches="tight", pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

    print("Saving heatmaps to:", heatmap_dir)
    print("Saving overlays to:", overlay_dir)

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]
        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=k)

        am_up = upsample_anomaly_map(anomaly_map, img_size)

        img_path = Path(path)
        base_name = img_path.stem
        label_str = f"label{label}"

        heatmap_path = heatmap_dir / f"{base_name}_{label_str}.png"
        overlay_path = overlay_dir / f"{base_name}_{label_str}.png"

        save_heatmap_and_overlay(str(img_path), am_up, img_size, heatmap_path, overlay_path)

    print("Done saving ALL baseline heatmaps and overlays")

    pixel_y_true: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]
        if label == 0:
            continue

        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=k)
        am_up = upsample_anomaly_map(anomaly_map, img_size)

        gt_mask = load_ground_truth_mask(path, data_root, class_name, img_size)
        if gt_mask is None:
            continue

        pixel_y_true.append(gt_mask.flatten())
        pixel_scores.append(am_up.flatten())

    if len(pixel_y_true) == 0:
        print("No ground-truth masks found for defective images.")
        return

    pixel_y_true_np = np.concatenate(pixel_y_true, axis=0)
    pixel_scores_np = np.concatenate(pixel_scores, axis=0)

    pixel_auc = roc_auc_score(pixel_y_true_np, pixel_scores_np)
    fpr, tpr, _ = roc_curve(pixel_y_true_np, pixel_scores_np)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{class_name} – Pixel-level ROC (kNN baseline)")
    plt.legend()
    plt.grid(True)
    plt.savefig(roc_dir / "pixel_roc.png", bbox_inches="tight", pad_inches=0)
    if show_plots:
        plt.show()
    plt.close()

    print(f"[{class_name}] Pixel-level ROC AUC: {pixel_auc:.4f}")


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    main()
