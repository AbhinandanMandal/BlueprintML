"""Atomic checkpoint I/O, resume-compatibility checks and model restoration.

Checkpoints are pickled with ``weights_only=False`` because they hold RNG states
and the config. Only load checkpoints you created yourself.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import TrainConfig
from .models import SimCLRModel, build_model
from .utils import get_environment_info, get_rng_state

CHECKPOINT_VERSION = 1

# Changing any of these mid-run would silently change the experiment.
MUST_MATCH = ("dataset", "train_subset_size", "subset_seed", "seed", "image_size", "encoder", "cifar_stem",
              "projection_hidden_dim", "projection_dim", "projection_bn", "temperature", "batch_size",
              "optimizer", "base_lr", "momentum", "weight_decay", "lars_trust_coefficient",
              "crop_scale_min", "crop_scale_max", "color_strength", "blur")
SCHEDULE_KEYS = ("epochs", "warmup_epochs")


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """Write to a temporary file then rename, so a crash never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_payload(model: SimCLRModel, optimizer, scheduler, scaler, cfg: TrainConfig, epoch: int,
                  global_step: int, best_metric: Optional[float], best_metric_name: str) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "encoder": model.encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "config": cfg.to_dict(),
        "seed": cfg.seed,
        "rng_state": get_rng_state(),
        "environment": get_environment_info(),
    }


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint {path} (corrupt or from an incompatible version)") from exc
    missing = {"encoder", "projector", "optimizer", "scheduler", "epoch", "global_step", "config"} - set(ckpt)
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing keys {sorted(missing)}")
    return ckpt


def check_resume_compatibility(ckpt_cfg: Dict[str, Any], cfg: TrainConfig, extend_schedule: bool) -> None:
    """Raise ``ValueError`` with an actionable message when resuming would change the experiment."""
    current = cfg.to_dict()
    diffs = [f"  {k}: checkpoint={ckpt_cfg.get(k)!r} vs now={current[k]!r}"
             for k in MUST_MATCH if k in ckpt_cfg and ckpt_cfg[k] != current[k]]
    if diffs:
        raise ValueError("Cannot resume: configuration differs from the checkpoint in fields that define the "
                         "experiment:\n" + "\n".join(diffs) + "\nUse the original settings, or start a new "
                         "run with a different --output-dir.")
    sched = [k for k in SCHEDULE_KEYS if k in ckpt_cfg and ckpt_cfg[k] != current[k]]
    if sched and not extend_schedule:
        raise ValueError(
            "Cannot resume: schedule differs from the checkpoint (" +
            ", ".join(f"{k}: {ckpt_cfg[k]!r} -> {current[k]!r}" for k in sched) +
            "). Pass --extend-schedule to continue with a new horizon (the cosine schedule is rebuilt for the "
            "new length, so the learning rate may jump at the resume point), or start a fresh run.")


def load_model_for_eval(path: str | Path, device: torch.device) -> tuple[SimCLRModel, TrainConfig]:
    """Rebuild the model from the checkpoint's own config and load encoder + projector weights."""
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = TrainConfig.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.encoder.load_state_dict(ckpt["encoder"])
    model.projector.load_state_dict(ckpt["projector"])
    return model.to(device).eval(), cfg
