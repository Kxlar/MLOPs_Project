import argparse
import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ApiSpec:
    endpoint: str
    form_field: str


def infer_label_from_path(path: Path) -> int | None:
    parts = {p.lower() for p in path.parts}
    if "good" in parts:
        return 0
    # MVTec convention: non-good folders in test are anomalies
    return 1


def iter_images(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API inference over a dataset folder and save heatmaps + scores.")

    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing images (recursively scanned)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write outputs")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--api_kind", choices=["bento", "fastapi"], default="bento")
    parser.add_argument("--endpoint", type=str, default=None, help="Override endpoint (default depends on api_kind)")
    parser.add_argument(
        "--form_field", type=str, default=None, help="Override multipart field (default depends on api_kind)"
    )

    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_heatmaps", action="store_true", help="Don't save decoded heatmaps")

    return parser.parse_args()


def resolve_api_spec(args: argparse.Namespace) -> ApiSpec:
    if args.api_kind == "bento":
        default = ApiSpec(endpoint="/predict", form_field="image")
    else:
        default = ApiSpec(endpoint="/predict", form_field="file")

    endpoint = args.endpoint or default.endpoint
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return ApiSpec(endpoint=endpoint, form_field=args.form_field or default.form_field)


def main() -> None:
    args = get_args()
    api = resolve_api_spec(args)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "scores.jsonl"

    url = f"http://{args.host}:{args.port}{api.endpoint}"
    images = list(iter_images(input_dir))
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"Found {len(images)} images under {input_dir}")
    print(f"Calling {url} (field: {api.form_field})")
    print(f"Writing results to {output_dir}")

    written = 0
    with open(scores_path, "a", encoding="utf-8") as scores_f:
        for img_path in images:
            rel = img_path.relative_to(input_dir)
            out_img_path = heatmaps_dir / rel.with_suffix(".png")
            out_img_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_img_path.exists() and not args.no_heatmaps:
                continue

            with open(img_path, "rb") as f:
                files = {api.form_field: f}
                resp = requests.post(url, files=files, timeout=args.timeout)

            record: dict = {
                "path": str(img_path.as_posix()),
                "relpath": str(rel.as_posix()),
                "status_code": resp.status_code,
            }

            if resp.status_code != 200:
                record["error"] = resp.text
                scores_f.write(json.dumps(record) + "\n")
                continue

            data = resp.json()
            record["anomaly_score"] = data.get("anomaly_score")
            record["is_anomaly"] = data.get("is_anomaly")
            record["label"] = infer_label_from_path(img_path)

            heatmap_b64 = data.get("heatmap_base64")
            if heatmap_b64 and not args.no_heatmaps:
                image_data = base64.b64decode(heatmap_b64)
                heatmap = Image.open(BytesIO(image_data))
                heatmap.save(out_img_path)

            scores_f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Done. Wrote {written} records to {scores_path}")


if __name__ == "__main__":
    main()
