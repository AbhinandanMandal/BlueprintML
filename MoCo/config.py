"""Single configuration object for every experiment knob, with validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_DATASETS = ("cifar10", "synthetic")
_BACKBONES = ("resnet18", "resnet50")
_SCHEDULES = ("none", "step", "cosine")
_OPTIMIZERS = ("sgd",)


@dataclass
class MoCoConfig:
    """All experiment settings.

    Defaults follow the MoCo v1 paper where the paper specifies a value
    (m=0.999, tau=0.07, dim=128, SGD lr=0.03 / momentum=0.9 / wd=1e-4,
    step decay x0.1 at 60% and 80% of training, i.e. epochs 120/160 of 200).
    Everything else is a Colab-scale engineering choice.
    """

    # ---- data -----------------------------------------------------------
    # "synthetic" is ONLY for checking mechanics when CIFAR-10 cannot be
    # downloaded. Never report accuracy on it as a research result.
    dataset: str = "cifar10"
    data_dir: str = "/content/data"
    subset_size: int = 1000          # number of images used for SSL training
    image_size: int = 32
    num_workers: int = 2

    # ---- model ----------------------------------------------------------
    backbone: str = "resnet18"
    cifar_stem: bool = True          # 3x3 stride-1 conv1, no maxpool (CIFAR adaptation)
    embedding_dim: int = 128
    queue_size: int = 256            # K, number of negative keys
    momentum: float = 0.999          # m, key-encoder momentum
    temperature: float = 0.07        # tau
    bn_groups: int = 1               # >1 simulates shuffling BN on one device (see README)

    # ---- optimisation ---------------------------------------------------
    batch_size: int = 32
    epochs: int = 1
    max_steps: Optional[int] = None  # hard cap on optimisation steps (smoke test)
    optimizer: str = "sgd"
    lr: float = 0.03
    sgd_momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_schedule: str = "none"        # none | step | cosine
    lr_milestones: Tuple[float, ...] = (0.6, 0.8)  # fractions of total epochs
    lr_gamma: float = 0.1
    amp: bool = False                # mixed precision (CUDA only)

    # ---- evaluation (labels are used ONLY here, never in SSL training) --
    eval_train_size: int = 5000      # labelled train images for kNN / linear probe
    eval_test_size: int = 10000      # test images used for reporting
    linear_epochs: int = 100
    linear_lr: float = 1e-3
    linear_weight_decay: float = 1e-5
    knn_k: int = 20

    # ---- bookkeeping ----------------------------------------------------
    seed: int = 42
    device: str = "auto"             # auto | cpu | cuda
    deterministic: bool = False
    log_every: int = 10
    checkpoint_dir: str = "/content/outputs/checkpoints"
    output_dir: str = "/content/outputs"
    checkpoint_every_epochs: int = 1

    # ------------------------------------------------------------------
    def validate(self) -> "MoCoConfig":
        """Raise ValueError with a clear message for any invalid setting."""
        errors = []
        if self.dataset not in _DATASETS:
            errors.append(f"dataset must be one of {_DATASETS}, got {self.dataset!r}")
        if self.backbone not in _BACKBONES:
            errors.append(f"backbone must be one of {_BACKBONES}, got {self.backbone!r}")
        if self.optimizer not in _OPTIMIZERS:
            errors.append(f"optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}")
        if self.lr_schedule not in _SCHEDULES:
            errors.append(f"lr_schedule must be one of {_SCHEDULES}, got {self.lr_schedule!r}")
        if self.device not in ("auto", "cpu", "cuda"):
            errors.append(f"device must be auto|cpu|cuda, got {self.device!r}")
        if self.batch_size < 2:
            errors.append("batch_size must be >= 2 (BatchNorm needs more than one sample)")
        if self.subset_size < self.batch_size:
            errors.append(
                f"subset_size ({self.subset_size}) must be >= batch_size ({self.batch_size}); "
                "the DataLoader drops the last incomplete batch"
            )
        if self.queue_size < self.batch_size:
            errors.append(
                f"queue_size ({self.queue_size}) must be >= batch_size ({self.batch_size})"
            )
        if self.embedding_dim < 1:
            errors.append("embedding_dim must be positive")
        if not (0.0 <= self.momentum < 1.0):
            errors.append(f"momentum must be in [0, 1), got {self.momentum}")
        if self.temperature <= 0:
            errors.append(f"temperature must be > 0, got {self.temperature}")
        if self.lr <= 0:
            errors.append(f"lr must be > 0, got {self.lr}")
        if self.bn_groups < 1 or self.batch_size % self.bn_groups != 0:
            errors.append("bn_groups must be >= 1 and divide batch_size evenly")
        elif self.batch_size // self.bn_groups < 2:
            errors.append("each BN group needs at least 2 samples (batch_size / bn_groups >= 2)")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            errors.append("max_steps must be >= 1 when set")
        if self.image_size < 16:
            errors.append("image_size must be >= 16")
        if self.num_workers < 0:
            errors.append("num_workers must be >= 0")
        if self.log_every < 1 or self.checkpoint_every_epochs < 1:
            errors.append("log_every and checkpoint_every_epochs must be >= 1")
        ms = tuple(self.lr_milestones)
        if any(not (0.0 < f < 1.0) for f in ms) or list(ms) != sorted(ms):
            errors.append("lr_milestones must be increasing fractions in (0, 1)")
        if self.amp and self.device == "cpu":
            errors.append("amp=True requires a CUDA device")
        if self.eval_train_size < 1 or self.eval_test_size < 1 or self.knn_k < 1:
            errors.append("eval_train_size, eval_test_size and knn_k must be positive")
        if errors:
            raise ValueError("Invalid MoCoConfig:\n  - " + "\n  - ".join(errors))
        return self

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lr_milestones"] = list(d["lr_milestones"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoCoConfig":
        d = dict(d)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "lr_milestones" in d:
            d["lr_milestones"] = tuple(d["lr_milestones"])
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> "MoCoConfig":
        """Load a .json or .yaml/.yml config file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency, preinstalled on Colab
            except ImportError as exc:
                raise ImportError("Reading YAML configs needs PyYAML: pip install pyyaml") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def with_overrides(self, **kwargs: Any) -> "MoCoConfig":
        return replace(self, **kwargs).validate()


def smoke_config(**overrides: Any) -> MoCoConfig:
    """Mode A: quick mechanics check (1,000 images, K=256, 20 steps)."""
    base = dict(subset_size=1000, batch_size=32, queue_size=256, epochs=1, max_steps=20,
                lr_schedule="none", log_every=5)
    base.update(overrides)
    return MoCoConfig(**base).validate()


def small_config(**overrides: Any) -> MoCoConfig:
    """Mode B: small CIFAR-10 study (5,000 images, K=4096, step schedule).

    ADAPTED configuration: the lr (0.03) is the paper's value for batch 256 and is
    deliberately NOT linearly scaled down for batch 64; the 'step' schedule keeps the
    paper's 60%/80% decay points but compresses them to the short run.
    """
    base = dict(subset_size=5000, batch_size=64, queue_size=4096, epochs=10, max_steps=None,
                lr_schedule="step", log_every=20)
    base.update(overrides)
    return MoCoConfig(**base).validate()
