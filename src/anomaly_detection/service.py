import bentoml
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
import io
import base64
import matplotlib.pyplot as plt


# Define the service ta mère
@bentoml.service(
    name="anomaly_detection_service", resources={"cpu": "4"}, traffic={"timeout": 60}
)
class AnomalyDetector:
    def __init__(self):
        # 1. Load Memory Bank (Standard PyTorch Tensor)
        self.device = torch.device("cpu")
        self.memory_bank_path = Path("./models/memory_bank.pt")
        self.onnx_model_path = Path("./models/dinov3_features.onnx")

        print(f"Loading Memory Bank from {self.memory_bank_path} ...")
        self.memory_bank = torch.load(self.memory_bank_path, map_location=self.device)

        # 2. Load ONNX Model
        print(f"Loading ONNX Session from {self.onnx_model_path} ...")
        self.ort_session = ort.InferenceSession(str(self.onnx_model_path))
        self.img_size = 224

    def transform_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image manually to match TorchVision transforms"""
        img = image.convert("RGB").resize((self.img_size, self.img_size))
        img_np = np.array(img).astype(np.float32) / 255.0

        # Normalize (Mean/Std for ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # Transpose to [C, H, W] and add Batch dim [1, C, H, W]
        img_np = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_np, axis=0)

    def compute_anomaly_map(self, features: np.ndarray, k: int = 10):
        """Re-implementation of model.py logic for ONNX outputs"""
        feat_t = torch.from_numpy(features).to(self.device)

        # features shape from ONNX: [1, C, Hf, Wf]
        _, C, Hf, Wf = feat_t.shape
        feat_t = feat_t.view(C, -1).T  # [N_patches, C]
        feat_t = F.normalize(feat_t, dim=1)

        # kNN Search
        dists = torch.cdist(feat_t, self.memory_bank)
        dists_sorted, _ = torch.sort(dists, dim=1)
        knn_dists = dists_sorted[:, :k]

        # Reshape back to map
        anomaly_score_patch = knn_dists.mean(dim=1)
        anomaly_map = anomaly_score_patch.view(Hf, Wf)

        return anomaly_map.cpu()

    def generate_heatmap_b64(
        self, anomaly_map: torch.Tensor, original_img: Image.Image
    ):
        # Upsample
        am = anomaly_map.unsqueeze(0).unsqueeze(0)
        am_up = F.interpolate(
            am,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Normalize 0-1
        am_up = (am_up - am_up.min()) / (am_up.max() - am_up.min() + 1e-8)
        am_np = am_up.numpy()

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(original_img.resize((self.img_size, self.img_size)))
        ax.imshow(am_np, cmap="jet", alpha=0.5)
        ax.axis("off")

        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @bentoml.api
    def predict(self, image: Image.Image) -> dict:
        # 1. Preprocess
        img_input = self.transform_image(image)  # Returns numpy [1, 3, 224, 224]

        # 2. ONNX Inference
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name

        # Run inference
        features = self.ort_session.run([output_name], {input_name: img_input})[0]

        # 3. Anomaly Logic
        anomaly_map = self.compute_anomaly_map(features)

        # 4. Metrics
        score = float(anomaly_map.max().item())
        threshold = 0.65

        # 5. Visuals
        heatmap = self.generate_heatmap_b64(anomaly_map, image)

        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": score > threshold,
            "heatmap_base64": heatmap,
        }
