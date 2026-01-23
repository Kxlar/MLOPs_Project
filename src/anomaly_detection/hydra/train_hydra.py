from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_detection.hydra.hydra_utils import cfg_to_namespace
from src.anomaly_detection.train import run

@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    args = cfg_to_namespace(cfg)
    run(args)


if __name__ == "__main__":
    main()
