from __future__ import annotations

import typer

from src.anomaly_detection.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_IMG_SIZE,
    DEFAULT_K,
    DEFAULT_WEIGHTS,
    run_baseline,
)

app = typer.Typer(help="Command-line interface for the anomaly detection baseline.")


@app.command()
def baseline(
    weights: str = typer.Option(
        DEFAULT_WEIGHTS,
        "--weights",
        help="Path to DINOv3 weights .pth file.",
    ),
    data_root: str = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root folder containing the MVTec-style dataset.",
    ),
    class_name: str = typer.Option(
        DEFAULT_CLASS_NAME,
        "--class-name",
        help="MVTec class to evaluate (e.g., carpet).",
    ),
    img_size: int = typer.Option(
        DEFAULT_IMG_SIZE,
        "--img-size",
        min=64,
        help="Image resolution fed to the ViT.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        min=1,
        help="Batch size for building the memory bank.",
    ),
    k: int = typer.Option(
        DEFAULT_K,
        "-k",
        min=1,
        help="Number of nearest neighbours for anomaly scoring.",
    ),
    show_plots: bool = typer.Option(
        False,
        "--show-plots/--no-show-plots",
        help="Display matplotlib windows in addition to saving them.",
    ),
) -> None:
    """Run the kNN baseline end-to-end: feature bank, scores, and visualizations."""

    run_baseline(
        weights=weights,
        data_root=data_root,
        class_name=class_name,
        img_size=img_size,
        batch_size=batch_size,
        k=k,
        show_plots=show_plots,
    )


def main() -> None:  # pragma: no cover - Typer injects main
    app()


if __name__ == "__main__":
    main()
