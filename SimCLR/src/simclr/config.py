"""Typed, validated configuration objects and a dataclass-driven CLI helper.

Precedence (lowest to highest): dataclass defaults < YAML file (``--config``) < CLI flags.
"""
from __future__ import annotations

import argparse
import dataclasses
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

T = TypeVar("T")

DATASETS = ("cifar10", "synthetic")
ENCODERS = ("resnet18", "resnet50")
OPTIMIZERS = ("sgd", "lars")

# Defaults per optimizer when base_lr / weight_decay are left unset.
# LARS values follow the SimCLR paper (lr = 0.3 x batch/256, wd = 1e-6).
# The SGD values are untuned practical defaults for small CIFAR runs.
DEFAULT_BASE_LR = {"sgd": 0.5, "lars": 0.3}
DEFAULT_WEIGHT_DECAY = {"sgd": 5e-4, "lars": 1e-6}


def _f(default: Any, help: str):
    return field(default=default, metadata={"help": help})


@dataclass
class TrainConfig:
    """Configuration of SimCLR pre-training."""

    # ---- data ----
    dataset: str = _f("cifar10", "Dataset: 'cifar10' or 'synthetic' (offline plumbing tests only).")
    data_dir: str = _f("./data", "Directory for dataset files.")
    output_dir: str = _f("./outputs/run", "Directory for checkpoints, logs and config dumps.")
    download: bool = _f(True, "Download CIFAR-10 if it is missing from --data-dir.")
    train_subset_size: int = _f(2000, "Number of training images used for pre-training (0 = all).")
    eval_subset_size: int = _f(1000, "Test images in the kNN monitor subset (0 = whole test split).")
    subset_seed: Optional[int] = _f(None, "Seed for subset selection (default: --seed).")
    num_workers: int = _f(2, "DataLoader worker processes.")
    image_size: int = _f(32, "Training/eval resolution. Values > 32 upsample CIFAR-10 (costly).")
    # ---- augmentation ----
    crop_scale_min: float = _f(0.08, "RandomResizedCrop minimum area fraction.")
    crop_scale_max: float = _f(1.0, "RandomResizedCrop maximum area fraction.")
    color_strength: float = _f(0.5, "Colour-distortion strength s (paper uses 0.5 for CIFAR-10).")
    blur: bool = _f(False, "Enable Gaussian blur (paper omits it for CIFAR-10).")
    # ---- model ----
    encoder: str = _f("resnet18", "Base encoder: 'resnet18' or 'resnet50'.")
    cifar_stem: bool = _f(True, "3x3 stride-1 stem and no max-pool (CIFAR adaptation).")
    projection_hidden_dim: int = _f(2048, "Hidden width of the projection head.")
    projection_dim: int = _f(128, "Output width of the projection head.")
    projection_bn: bool = _f(True, "BatchNorm after the hidden projection layer.")
    temperature: float = _f(0.5, "NT-Xent temperature tau.")
    # ---- optimisation ----
    epochs: int = _f(3, "Total number of pre-training epochs.")
    batch_size: int = _f(64, "Images per step (the loss sees 2x this many views).")
    optimizer: str = _f("sgd", "'sgd' (momentum) or 'lars'.")
    base_lr: Optional[float] = _f(None, "Learning rate for batch 256; actual lr = base_lr * batch/256.")
    momentum: float = _f(0.9, "Momentum.")
    weight_decay: Optional[float] = _f(None, "Weight decay (default depends on optimizer).")
    lars_trust_coefficient: float = _f(0.001, "LARS trust coefficient eta.")
    warmup_epochs: float = _f(1.0, "Linear warm-up length in epochs (may be fractional).")
    grad_clip: Optional[float] = _f(None, "Global grad-norm clip (disabled by default).")
    amp: bool = _f(True, "Mixed precision (only used on CUDA).")
    # ---- run control ----
    seed: int = _f(42, "Global random seed.")
    deterministic: bool = _f(False, "Request deterministic kernels (slower; not a perfect guarantee).")
    device: str = _f("auto", "'auto', 'cpu' or 'cuda'.")
    save_every: int = _f(1, "Write last_checkpoint.pt every N epochs (and at the end).")
    eval_every: int = _f(1, "Run the kNN monitor every N epochs (0 = disabled).")
    resume: Optional[str] = _f(None, "Checkpoint path to resume from, or 'auto' for <output_dir>/last_checkpoint.pt.")
    extend_schedule: bool = _f(False, "Allow --epochs to differ from the checkpoint's schedule when resuming.")

    # ------------------------------------------------------------------
    @property
    def lr(self) -> float:
        base = self.base_lr if self.base_lr is not None else DEFAULT_BASE_LR[self.optimizer]
        return base * self.batch_size / 256.0

    @property
    def wd(self) -> float:
        return self.weight_decay if self.weight_decay is not None else DEFAULT_WEIGHT_DECAY[self.optimizer]

    @property
    def effective_subset_seed(self) -> int:
        return self.seed if self.subset_seed is None else self.subset_seed

    def validate(self) -> "TrainConfig":
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"Invalid configuration: {msg}")

        need(self.dataset in DATASETS, f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        need(self.encoder in ENCODERS, f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        need(self.optimizer in OPTIMIZERS, f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        need(self.device in ("auto", "cpu", "cuda"), "device must be 'auto', 'cpu' or 'cuda'")
        need(self.batch_size >= 2, "batch_size must be >= 2 (NT-Xent needs negatives)")
        need(self.epochs >= 1, "epochs must be >= 1")
        need(self.temperature > 0, "temperature must be > 0")
        need(self.projection_dim >= 1 and self.projection_hidden_dim >= 1, "projection dims must be >= 1")
        need(self.train_subset_size >= 0 and self.eval_subset_size >= 0, "subset sizes must be >= 0")
        need(self.train_subset_size == 0 or self.train_subset_size >= self.batch_size,
             "train_subset_size must be 0 (all) or >= batch_size (drop_last would give zero batches)")
        need(0 < self.crop_scale_min <= self.crop_scale_max <= 1.0, "need 0 < crop_scale_min <= crop_scale_max <= 1")
        need(0.0 <= self.color_strength <= 2.5, "color_strength must be in [0, 2.5] (hue jitter must stay <= 0.5)")
        need(self.image_size >= 8, "image_size must be >= 8")
        need(self.warmup_epochs >= 0, "warmup_epochs must be >= 0")
        need(self.warmup_epochs < self.epochs, "warmup_epochs must be smaller than epochs")
        need(self.num_workers >= 0, "num_workers must be >= 0")
        need(self.save_every >= 1, "save_every must be >= 1")
        need(self.eval_every >= 0, "eval_every must be >= 0")
        need(self.grad_clip is None or self.grad_clip > 0, "grad_clip must be positive when set")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LinearEvalConfig:
    """Configuration of the frozen-encoder linear evaluation."""

    checkpoint: str = _f("", "Path to a pre-training checkpoint (required).")
    dataset: Optional[str] = _f(None, "Dataset (default: the one used for pre-training).")
    data_dir: str = _f("./data", "Directory for dataset files.")
    output_dir: str = _f("./outputs/linear_eval", "Where results are written.")
    download: bool = _f(True, "Download CIFAR-10 if missing.")
    train_subset_size: int = _f(0, "Labelled training images for the probe (0 = whole train split).")
    test_subset_size: int = _f(0, "Test images (0 = whole official test split).")
    subset_seed: int = _f(42, "Seed for subset selection and the val split.")
    val_fraction: float = _f(0.1, "Fraction of labelled train images held out for monitoring.")
    epochs: int = _f(100, "Linear-classifier epochs.")
    batch_size: int = _f(512, "Linear-classifier batch size.")
    lr: float = _f(1e-3, "Adam learning rate.")
    weight_decay: float = _f(0.0, "Adam weight decay.")
    standardize: bool = _f(True, "Standardise features with train-split statistics (still a linear probe).")
    num_workers: int = _f(2, "DataLoader workers for feature extraction.")
    device: str = _f("auto", "'auto', 'cpu' or 'cuda'.")
    seed: int = _f(42, "Global seed.")

    def validate(self) -> "LinearEvalConfig":
        if not self.checkpoint:
            raise ValueError("Invalid configuration: --checkpoint is required")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("Invalid configuration: val_fraction must be in [0, 1)")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Invalid configuration: epochs and batch_size must be >= 1")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------
def _unwrap_optional(tp: Any) -> Any:
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def add_config_arguments(parser: argparse.ArgumentParser, cls: Type[T]) -> None:
    """Add one ``--kebab-case`` flag per dataclass field (defaults suppressed)."""
    hints = typing.get_type_hints(cls)
    for f in fields(cls):
        flag = "--" + f.name.replace("_", "-")
        tp = _unwrap_optional(hints[f.name])
        help_text = f"{f.metadata.get('help', '')} (default: {f.default})"
        if tp is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                                default=argparse.SUPPRESS, help=help_text)
        else:
            parser.add_argument(flag, dest=f.name, type=tp, default=argparse.SUPPRESS, help=help_text)


def build_config(cls: Type[T], args: argparse.Namespace, yaml_path: Optional[str] = None) -> T:
    """Merge defaults, an optional YAML file and parsed CLI flags into a validated config."""
    values: Dict[str, Any] = {}
    if yaml_path:
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        loaded = yaml.safe_load(path.read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(loaded) - known)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {unknown}")
        values.update(loaded)
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    values.update(cli)
    cfg = cls(**values)
    return cfg.validate()
