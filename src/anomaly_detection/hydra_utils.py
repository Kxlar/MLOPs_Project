from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, OmegaConf


def cfg_to_namespace(cfg: DictConfig) -> SimpleNamespace:
    """Convert a Hydra DictConfig to an argparse-like object.

    Your existing code expects attribute access (args.foo). This helper keeps
    that interface so we can reuse build_dataloaders(...) etc.
    """

    data: Any = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected cfg to convert to dict, got {type(data)}")
    return SimpleNamespace(**data)
