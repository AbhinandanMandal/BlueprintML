"""Training-curve plots saved to the output directory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_training(history: List[Dict[str, Any]], out_dir: str, tag: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [r["loss"] for r in history])
    ax.set(xlabel="step", ylabel="InfoNCE loss", title="Contrastive training loss")
    fig.tight_layout(); fig.savefig(Path(out_dir) / f"{tag}_loss.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [r["lr"] for r in history])
    ax.set(xlabel="step", ylabel="learning rate", title="Learning-rate schedule")
    fig.tight_layout(); fig.savefig(Path(out_dir) / f"{tag}_lr.png", dpi=120); plt.close(fig)
