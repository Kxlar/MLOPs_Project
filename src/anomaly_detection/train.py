# train.py
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc

from data import build_dataloaders, load_ground_truth_mask
from model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    build_memory_bank,
    compute_anomaly_map,
    reduce_anomaly_map,
    upsample_anomaly_map,
)

# =======================
# CONFIG (your paths)
# =======================
DINOV3_LOCATION = r"C:\Users\snehi\Documents\Deep Learning\dinov3"
WEIGHTS = r"C:\Users\snehi\Documents\Deep Learning\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

DATA_ROOT = r"C:\Users\snehi\Documents\Deep Learning\mvtec_anomaly_detection"
CLASS_NAME = "carpet"
IMG_SIZE = 224
BATCH_SIZE = 8
K = 10

RESULTS_ROOT = Path("results_baseline") / CLASS_NAME
HEATMAP_DIR = RESULTS_ROOT / "heatmaps"
OVERLAY_DIR = RESULTS_ROOT / "overlays"
ROC_DIR = RESULTS_ROOT / "roc"

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
ROC_DIR.mkdir(parents=True, exist_ok=True)


def save_heatmap_and_overlay(img_path: str, am_up: np.ndarray, img_size: int, out_heatmap: Path, out_overlay: Path):
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


def main():
    print("DATA_ROOT:", DATA_ROOT)
    print("CLASS_NAME:", CLASS_NAME)
    print("DINOv3_REPO:", DINOV3_LOCATION)
    print("DINOv3_WEIGHTS:", WEIGHTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------
    # Data
    # -----------------------
    train_dataset, test_dataset, train_loader = build_dataloaders(
        DATA_ROOT, CLASS_NAME, IMG_SIZE, batch_size=BATCH_SIZE
    )
    print("Train images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    test_labels = np.array(test_dataset.labels)
    n_good = int(np.sum(test_labels == 0))
    n_def = int(np.sum(test_labels == 1))
    print(f"Test set stats for '{CLASS_NAME}': good={n_good}, defective={n_def}, total={len(test_dataset)}")

    # -----------------------
    # Model + feature extractor
    # -----------------------
    dinov3_model = load_dinov3(DINOV3_LOCATION, WEIGHTS, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)

    # -----------------------
    # Memory bank
    # -----------------------
    memory_bank = build_memory_bank(feature_extractor, train_loader, device)
    print("Memory bank shape:", tuple(memory_bank.shape))

    # -----------------------
    # Image-level evaluation + timing
    # -----------------------
    y_true = []
    y_score_max = []
    y_score_mean = []
    inference_times = []

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]

        start = time.time()
        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=K)
        end = time.time()

        inference_times.append(end - start)

        y_true.append(label)
        y_score_max.append(reduce_anomaly_map(anomaly_map, mode="max"))
        y_score_mean.append(reduce_anomaly_map(anomaly_map, mode="mean"))

    y_true = np.array(y_true)
    y_score_max = np.array(y_score_max)
    y_score_mean = np.array(y_score_mean)

    auc_max = roc_auc_score(y_true, y_score_max)
    auc_mean = roc_auc_score(y_true, y_score_mean)

    avg_ms = float(np.mean(inference_times)) * 1000
    std_ms = float(np.std(inference_times)) * 1000

    print(f"[{CLASS_NAME}] ROC AUC (max over patches):  {auc_max:.4f}")
    print(f"[{CLASS_NAME}] ROC AUC (mean over patches): {auc_mean:.4f}")
    print(f"[{CLASS_NAME}] Avg inference time: {avg_ms:.2f} ms (std {std_ms:.2f} ms, N={len(inference_times)})")

    # histogram
    plt.hist(y_score_max[y_true == 0], bins=20, alpha=0.6, label="good")
    plt.hist(y_score_max[y_true == 1], bins=20, alpha=0.6, label="defective")
    plt.legend()
    plt.title(f"{CLASS_NAME} - Baseline image anomaly scores (max)")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.savefig(ROC_DIR / "image_score_hist_max.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    # -----------------------
    # Save heatmaps + overlays
    # -----------------------
    print("Saving heatmaps to:", HEATMAP_DIR)
    print("Saving overlays to:", OVERLAY_DIR)

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]
        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=K)

        am_up = upsample_anomaly_map(anomaly_map, IMG_SIZE)

        img_path = Path(path)
        base_name = img_path.stem
        label_str = f"label{label}"

        heatmap_path = HEATMAP_DIR / f"{base_name}_{label_str}.png"
        overlay_path = OVERLAY_DIR / f"{base_name}_{label_str}.png"

        save_heatmap_and_overlay(str(img_path), am_up, IMG_SIZE, heatmap_path, overlay_path)

    print("Done saving ALL baseline heatmaps and overlays")

    # -----------------------
    # Pixel-level ROC (only defective images)
    # -----------------------
    pixel_y_true = []
    pixel_scores = []

    for i in range(len(test_dataset)):
        img_t, label, path = test_dataset[i]
        if label == 0:
            continue

        anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=K)
        am_up = upsample_anomaly_map(anomaly_map, IMG_SIZE)

        gt_mask = load_ground_truth_mask(path, DATA_ROOT, CLASS_NAME, IMG_SIZE)
        if gt_mask is None:
            continue

        pixel_y_true.append(gt_mask.flatten())
        pixel_scores.append(am_up.flatten())

    if len(pixel_y_true) == 0:
        print("No ground-truth masks found for defective images.")
        return

    pixel_y_true = np.concatenate(pixel_y_true, axis=0)
    pixel_scores = np.concatenate(pixel_scores, axis=0)

    pixel_auc = roc_auc_score(pixel_y_true, pixel_scores)
    fpr, tpr, _ = roc_curve(pixel_y_true, pixel_scores)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{CLASS_NAME} – Pixel-level ROC (kNN baseline)")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_DIR / "pixel_roc.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    print(f"[{CLASS_NAME}] Pixel-level ROC AUC: {pixel_auc:.4f}")


if __name__ == "__main__":
    main()
