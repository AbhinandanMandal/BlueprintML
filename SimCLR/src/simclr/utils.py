"""Seeding, device selection, RNG capture, environment info and small I/O helpers."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices).

    ``deterministic=True`` requests deterministic kernels. Some CUDA ops still
    have no deterministic implementation (``warn_only`` is used), so bit-exact
    reproducibility is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available. "
                           "In Colab choose Runtime > Change runtime type > GPU.")
    return torch.device(name)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        saved = [s.cpu() for s in state["cuda"]]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved)


def get_environment_info() -> Dict[str, Any]:
    import torchvision

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


class CSVLogger:
    """Append-only CSV writer with a fixed header (safe across resumes)."""

    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open() as fh:
                header = next(csv.reader(fh), [])
            if header != self.fieldnames:
                raise ValueError(f"{self.path} has columns {header}, expected {self.fieldnames}. "
                                 "Use a fresh --output-dir.")
        else:
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writerow(
                {k: ("" if row.get(k) is None else row.get(k)) for k in self.fieldnames})
