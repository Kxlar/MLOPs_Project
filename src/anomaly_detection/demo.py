import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.use("Agg")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side comparison of two histogram images.")
    p.add_argument("--hist1", type=str, required=True, help="Path to first histogram PNG")
    p.add_argument("--hist2", type=str, required=True, help="Path to second histogram PNG")
    p.add_argument("--title1", type=str, default="Before (original memory bank)")
    p.add_argument("--title2", type=str, default="After (drifted memory bank)")
    p.add_argument("--out", type=str, default="./results/hist_comparison.png")
    return p.parse_args()


def main() -> None:
    args = get_args()

    hist1 = Image.open(Path(args.hist1))
    hist2 = Image.open(Path(args.hist2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(hist1)
    axes[0].set_title(args.title1)
    axes[0].axis("off")

    axes[1].imshow(hist2)
    axes[1].set_title(args.title2)
    axes[1].axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Saved comparison figure to: {out}")


if __name__ == "__main__":
    main()
