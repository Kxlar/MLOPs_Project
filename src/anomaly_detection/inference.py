import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.data import MVTecDataset, build_transform
from src.anomaly_detection.model import (
    load_dinov3,
    DINOv3FeatureExtractor,
    compute_anomaly_map,
    reduce_anomaly_map,
    upsample_anomaly_map,
)


def get_args():
    parser = argparse.ArgumentParser(description="Inference: Generate Heatmaps")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--memory_bank_path", type=str, required=True, help="Path to .pt memory bank")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output folder name (defaults to --class_name)",
    )

    parser.add_argument("--split", choices=["train", "test"], default="test")

    parser.add_argument("--save_heatmaps", action="store_true", default=True)
    parser.add_argument("--save_overlays", action="store_true", default=True)
    parser.add_argument("--heatmaps_only", action="store_true", help="Convenience flag: disable overlay saving")
    parser.add_argument(
        "--scores_jsonl",
        type=str,
        default=None,
        help="If set, write per-image anomaly scores + labels as JSONL",
    )

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k", type=int, default=10, help="Top-k neighbors")

    # Argparse compatibility for data.py
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug_types", nargs="+", default=[])

    return parser.parse_args()


def save_heatmap_and_overlay(img_path, am_up, img_size, out_heatmap, out_overlay):
    # Normalize 0-1
    am_norm = (am_up - am_up.min()) / (am_up.max() - am_up.min() + 1e-8)

    orig_img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    orig_np = np.array(orig_img)

    # Save heatmap
    plt.figure()
    plt.axis("off")
    plt.imshow(am_norm, cmap="jet")
    plt.tight_layout(pad=0)
    plt.savefig(out_heatmap, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save overlay
    plt.figure()
    plt.axis("off")
    plt.imshow(orig_np)
    plt.imshow(am_norm, cmap="jet", alpha=0.5)
    plt.tight_layout(pad=0)
    plt.savefig(out_overlay, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    args = get_args()
    run(args)


def run(args) -> None:
    """Run inference and save heatmaps/overlays.

    Kept as a separate function so Hydra can call the same logic.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.heatmaps_only:
        args.save_overlays = False

    # Setup Paths
    results_root = Path(args.output_dir) / (args.output_name or args.class_name)
    results_root.mkdir(parents=True, exist_ok=True)

    heatmap_dir = results_root / "heatmaps"
    overlay_dir = results_root / "overlays"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    scores_f = None
    if args.scores_jsonl:
        scores_path = Path(args.scores_jsonl)
    else:
        scores_path = None

    # 1. Load Data
    transform = build_transform(args.img_size, args=None, is_saving=False)
    dataset = MVTecDataset(args.data_root, args.class_name, split=args.split, transform=transform)
    print(f"{args.split.capitalize()} images: {len(dataset)}")

    # 2. Load Model & Memory Bank
    dinov3_model = load_dinov3(args.weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval().to(device)

    print(f"Loading memory bank from {args.memory_bank_path}...")
    memory_bank = torch.load(args.memory_bank_path, map_location=device)

    # 3. Inference Loop
    print("Starting inference...")
    if scores_path is not None:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_f = open(scores_path, "w", encoding="utf-8")

    try:
        for i in range(len(dataset)):
            img_t, label, path = dataset[i]

            anomaly_map = compute_anomaly_map(img_t, feature_extractor, memory_bank, k=args.k)
            anomaly_score = reduce_anomaly_map(anomaly_map, mode="max")
            am_up = upsample_anomaly_map(anomaly_map, args.img_size)

            img_path = Path(path)
            base_name = img_path.stem
            label_str = "defect" if label == 1 else "good"

            if scores_f is not None:
                import json

                scores_f.write(
                    json.dumps(
                        {
                            "path": str(img_path.as_posix()),
                            "label": int(label),
                            "anomaly_score": float(anomaly_score),
                        }
                    )
                    + "\n"
                )

            if args.save_heatmaps or args.save_overlays:
                out_heatmap = heatmap_dir / f"{base_name}_{label_str}.png" if args.save_heatmaps else None
                out_overlay = overlay_dir / f"{base_name}_{label_str}.png" if args.save_overlays else None

                if out_heatmap is None:
                    # Still need a path for the helper, but don't write it
                    out_heatmap = heatmap_dir / f"{base_name}_{label_str}.png"
                if out_overlay is None:
                    out_overlay = overlay_dir / f"{base_name}_{label_str}.png"

                if not args.save_heatmaps and not args.save_overlays:
                    pass
                elif args.save_heatmaps and args.save_overlays:
                    save_heatmap_and_overlay(
                        str(img_path),
                        am_up,
                        args.img_size,
                        out_heatmap,
                        out_overlay,
                    )
                elif args.save_heatmaps and not args.save_overlays:
                    # Save only heatmap
                    plt.figure()
                    plt.axis("off")
                    am_norm = (am_up - am_up.min()) / (am_up.max() - am_up.min() + 1e-8)
                    plt.imshow(am_norm, cmap="jet")
                    plt.tight_layout(pad=0)
                    plt.savefig(out_heatmap, bbox_inches="tight", pad_inches=0)
                    plt.close()
                elif not args.save_heatmaps and args.save_overlays:
                    # Save only overlay
                    plt.figure()
                    plt.axis("off")
                    am_norm = (am_up - am_up.min()) / (am_up.max() - am_up.min() + 1e-8)
                    orig_img = Image.open(img_path).convert("RGB").resize((args.img_size, args.img_size))
                    orig_np = np.array(orig_img)
                    plt.imshow(orig_np)
                    plt.imshow(am_norm, cmap="jet", alpha=0.5)
                    plt.tight_layout(pad=0)
                    plt.savefig(out_overlay, bbox_inches="tight", pad_inches=0)
                    plt.close()

            if i % 10 == 0:
                print(f"Processed {i}/{len(dataset)}")
    finally:
        if scores_f is not None:
            scores_f.close()

    print(f"Done. Results saved in {results_root}")
    if scores_path is not None:
        print(f"Scores saved in {scores_path}")


if __name__ == "__main__":
    main()
