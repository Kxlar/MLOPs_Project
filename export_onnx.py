import torch
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from src.anomaly_detection.model import load_dinov3, DINOv3FeatureExtractor


def export_model(weights_path, output_path="./models/dinov3_features.onnx"):
    device = torch.device("cpu")  # ONNX export is typically done on CPU

    # 1. Load the model
    print(f"Loading weights from {weights_path}...")
    dinov3_model = load_dinov3(weights_path, device)
    feature_extractor = DINOv3FeatureExtractor(dinov3_model).eval()

    # Convert all model weights to standard float32 to avoid bfloat16 errors in ONNX
    feature_extractor = feature_extractor.float()

    # 2. Create dummy input (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # 3. Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        feature_extractor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Higher opset for better Transformer support
        do_constant_folding=True,
        input_names=["input"],
        output_names=["feature_map"],
        dynamic_axes={"input": {0: "batch_size"}, "feature_map": {0: "batch_size"}},
    )
    print("Export complete.")


if __name__ == "__main__":
    WEIGHTS = "./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    export_model(WEIGHTS)
