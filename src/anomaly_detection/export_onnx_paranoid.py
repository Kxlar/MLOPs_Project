import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.anomaly_detection.model import load_dinov3, DINOv3FeatureExtractor


def recursive_cast_attributes(module):
    """
    Forcefully updates any 'dtype' attributes in the model config
    that might be lingering as bfloat16.
    """
    for name, child in module.named_children():
        if hasattr(child, "dtype"):
            child.dtype = torch.float32
        recursive_cast_attributes(child)


def export_model(weights_path, output_path="./models/dinov3_features.onnx"):
    # 1. Cleanup Old File
    if os.path.exists(output_path):
        print(f"[-] Deleting old ONNX file: {output_path}")
        os.remove(output_path)

    device = torch.device("cpu")

    # 2. Load and Cast Model
    print(f"Loading weights from {weights_path}...")
    dinov3_model = load_dinov3(weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval()

    # Force Weights to Float32
    print("[-] Casting model weights to Float32...")
    feature_extractor = feature_extractor.float()

    # Force Attributes to Float32 (The "Paranoid" Step)
    recursive_cast_attributes(feature_extractor)

    # 3. Create Float32 Dummy Input
    dummy_input = torch.randn(1, 3, 224, 224, device=device).float()

    # 4. Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        feature_extractor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Downgrade to 14 (More stable for standard Float32 than 17)
        do_constant_folding=True,
        input_names=["input"],
        output_names=["feature_map"],
        dynamic_axes={"input": {0: "batch_size"}, "feature_map": {0: "batch_size"}},
    )

    # 5. Verify
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[+] Success! New ONNX file created: {size_mb:.2f} MB")
    else:
        print("[!] Error: ONNX file was not created.")


if __name__ == "__main__":
    WEIGHTS = "./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    export_model(WEIGHTS)
