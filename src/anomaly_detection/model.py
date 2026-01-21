# model.py
from pathlib import Path

import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.anomaly_detection.vision_transformer import vit_base


def load_dinov3(weights_path: str, device: torch.device):
    """
    Loads DINOv3 using the Meta DinoVisionTransformer architecture.
    """
    # Use the exact argument names from DinoVisionTransformer.__init__
    model = vit_base(
        patch_size=16,
        img_size=224,
        layerscale_init=1.0,
        n_storage_tokens=4,
    )

    state_dict = torch.load(weights_path, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "teacher" in state_dict:
        state_dict = state_dict["teacher"]

    # Filter out technical buffers that cause RuntimeError in strict mode
    # These are specific to DINOv3's attention implementation
    state_dict = {k: v for k, v in state_dict.items() if "bias_mask" not in k}

    # Loading with strict=False is recommended to ignore non-essential buffers
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loading status: {msg}")

    model.eval().to(device)
    return model


class DINOv3FeatureExtractor(nn.Module):
    """
    Transforms input from [B,3,H,W] -> [B,D,Hf,Wf]
    """

    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.dino.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, D]
        B, N, D = patch_tokens.shape

        side = int(N**0.5)
        assert side * side == N, f"Patch count {N} is not a perfect square"
        Hf = Wf = side

        patch_tokens = patch_tokens.view(B, Hf, Wf, D)
        patch_map = patch_tokens.permute(0, 3, 1, 2).contiguous()  # [B, D, Hf, Wf]
        return patch_map


@torch.no_grad()
def build_memory_bank(feature_extractor: nn.Module, train_loader, device: torch.device):
    """
    Builds memory bank from train images (good only).
    Returns tensor [N_mem, C] on device.
    """
    memory_bank = []

    for imgs, labels, paths in train_loader:
        imgs = imgs.to(device)
        feats = feature_extractor(imgs)  # [B,C,Hf,Wf]
        B, C, Hf, Wf = feats.shape

        feats = feats.view(B, C, -1)  # [B,C,N]
        feats = feats.permute(0, 2, 1)  # [B,N,C]
        feats = feats.reshape(-1, C)  # [B*N,C]
        feats = F.normalize(feats, dim=1)

        memory_bank.append(feats.cpu())

    memory_bank = torch.cat(memory_bank, dim=0)  # [N_mem, C]
    memory_bank = memory_bank.to(device)  # move once
    return memory_bank


@torch.no_grad()
def compute_anomaly_map(
    img_t: torch.Tensor,
    feature_extractor: nn.Module,
    memory_bank: torch.Tensor,
    k: int = 10,
):
    """
    img_t: [3,H,W] (already transformed)
    memory_bank: [N_mem,C] on same device
    returns: anomaly_map torch.Tensor [Hf,Wf] on CPU
    """
    img_t_batch = img_t.unsqueeze(0).to(memory_bank.device)

    feat = feature_extractor(img_t_batch)  # [1,C,Hf,Wf]
    _, C, Hf, Wf = feat.shape

    feat = feat.view(C, -1).T
    feat = F.normalize(feat, dim=1)

    dists = torch.cdist(feat, memory_bank)
    dists_sorted, _ = torch.sort(dists, dim=1)
    knn_dists = dists_sorted[:, :k]
    anomaly_score_patch = knn_dists.mean(dim=1)

    anomaly_map = anomaly_score_patch.view(Hf, Wf)
    return anomaly_map.cpu()


def reduce_anomaly_map(anomaly_map: torch.Tensor, mode: str = "max") -> float:
    if mode == "max":
        return float(anomaly_map.max().item())
    if mode == "mean":
        return float(anomaly_map.mean().item())
    raise ValueError(f"Unknown mode: {mode}")


def upsample_anomaly_map(anomaly_map: torch.Tensor, img_size: int) -> np.ndarray:
    """
    anomaly_map: torch.Tensor [Hf,Wf] CPU
    returns: numpy array [img_size,img_size]
    """
    am = anomaly_map.unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
    am_up = F.interpolate(
        am,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return am_up.squeeze().cpu().numpy()
