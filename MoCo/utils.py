"""Reproducibility, device selection, environment reporting and CSV helpers."""

from __future__ import annotations

import csv
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torchvision

from .config import MoCoConfig


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    deterministic=True forces deterministic cuDNN kernels (slower). Bitwise
    reproducibility across different hardware / library versions is NOT promised.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy/random inside each DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(requested: str = "auto") -> torch.device:
    """Resolve the device; fail early with an explanation if CUDA is requested but missing."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "In Colab choose Runtime > Change runtime type > GPU, then restart the "
                "runtime. Use device='auto' or 'cpu' to run without a GPU."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_report(cfg: Optional[MoCoConfig] = None) -> Dict[str, Any]:
    """Collect (and return) Test 1 information."""
    device = get_device(cfg.device if cfg else "auto")
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "subset_size": cfg.subset_size if cfg else None,
    }
    return report


def print_config(cfg: MoCoConfig) -> None:
    print("Configuration:")
    print(json.dumps(cfg.to_dict(), indent=2))


def peak_gpu_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("")
        return
    fieldnames: List[str] = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> List[Dict[str, float]]:
    with open(path, newline="") as f:
        out = []
        for row in csv.DictReader(f):
            out.append({k: (float(v) if v not in ("", "None") else float("nan")) for k, v in row.items()})
        return out
