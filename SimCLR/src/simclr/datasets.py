"""Dataset loading, deterministic subsetting and SimCLR/labelled dataset wrappers.

Self-supervised pre-training uses :class:`TransformDataset` with
``return_labels=False``: the label is dropped inside ``__getitem__``, so the
training loop cannot accidentally use it.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .augmentations import (CIFAR10_MEAN, CIFAR10_STD, TwoViewTransform, build_eval_transform,
                            build_train_transform)
from .config import TrainConfig

NUM_CLASSES = 10
SYNTHETIC_TRAIN_SIZE = 10_000
SYNTHETIC_TEST_SIZE = 2_000


class SyntheticImageDataset(Dataset):
    """Class-dependent colour images, generated offline and deterministically.

    ONLY for testing the pipeline where CIFAR-10 cannot be downloaded. It says
    nothing about representation quality on real data.
    """

    def __init__(self, size: int, seed: int, image_size: int = 32, num_classes: int = NUM_CLASSES) -> None:
        rng = np.random.default_rng(seed)
        palette = np.random.default_rng(12345).integers(30, 225, size=(num_classes, 3))
        self.targets = rng.integers(0, num_classes, size=size).tolist()
        yy, xx = np.mgrid[0:image_size, 0:image_size] / float(image_size)
        data = np.empty((size, image_size, image_size, 3), dtype=np.uint8)
        for i, label in enumerate(self.targets):
            gradient = (xx * (label % 3 - 1) + yy * (label % 4 - 1.5))[..., None] * 40
            img = palette[label] + gradient + rng.normal(0, 20, size=(image_size, image_size, 3))
            data[i] = np.clip(img, 0, 255).astype(np.uint8)
        self.data = data

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        return Image.fromarray(self.data[index]), self.targets[index]


def load_base_dataset(name: str, data_dir: str, train: bool, download: bool = True) -> Dataset:
    """Return a dataset yielding ``(PIL.Image, label)`` with no transform applied."""
    if name == "synthetic":
        return SyntheticImageDataset(SYNTHETIC_TRAIN_SIZE if train else SYNTHETIC_TEST_SIZE,
                                     seed=0 if train else 1)
    if name == "cifar10":
        from torchvision.datasets import CIFAR10
        try:
            return CIFAR10(root=data_dir, train=train, download=download)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load CIFAR-10 from {data_dir!r} (download={download}). Check network access "
                "or place the extracted 'cifar-10-batches-py' folder inside the data directory.") from exc
    raise ValueError(f"Unknown dataset {name!r}")


def deterministic_subset_indices(total: int, subset_size: int, seed: int) -> np.ndarray:
    """Sorted random subset of ``range(total)``; ``subset_size == 0`` selects everything."""
    if subset_size < 0:
        raise ValueError("subset_size must be >= 0")
    if subset_size == 0 or subset_size == total:
        return np.arange(total)
    if subset_size > total:
        raise ValueError(f"Requested subset of {subset_size} from only {total} samples")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=subset_size, replace=False))


class TransformDataset(Dataset):
    """Applies ``transform`` to the PIL image of a base dataset.

    ``return_labels=False`` yields only the transformed image(s) (self-supervised);
    ``True`` yields ``(x, label)`` (downstream evaluation only).
    """

    def __init__(self, base: Dataset, transform: Callable, return_labels: bool) -> None:
        self.base = base
        self.transform = transform
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        out = self.transform(image)
        return (out, label) if self.return_labels else out


def build_subset(name: str, data_dir: str, train: bool, subset_size: int, seed: int,
                 download: bool = True) -> Subset:
    base = load_base_dataset(name, data_dir, train=train, download=download)
    return Subset(base, deterministic_subset_indices(len(base), subset_size, seed).tolist())


def build_pretrain_dataset(cfg: TrainConfig) -> TransformDataset:
    """Unlabelled two-view dataset over a deterministic subset of the TRAIN split."""
    subset = build_subset(cfg.dataset, cfg.data_dir, True, cfg.train_subset_size,
                          cfg.effective_subset_seed, cfg.download)
    train_tf = build_train_transform(cfg.image_size, (cfg.crop_scale_min, cfg.crop_scale_max),
                                     cfg.color_strength, cfg.blur, mean=CIFAR10_MEAN, std=CIFAR10_STD)
    return TransformDataset(subset, TwoViewTransform(train_tf), return_labels=False)


def build_labeled_dataset(name: str, data_dir: str, train: bool, subset_size: int, seed: int,
                          image_size: int, download: bool = True) -> TransformDataset:
    """Labelled dataset with the deterministic eval transform (kNN / linear probe)."""
    subset = build_subset(name, data_dir, train, subset_size, seed, download)
    return TransformDataset(subset, build_eval_transform(image_size, CIFAR10_MEAN, CIFAR10_STD),
                            return_labels=True)


def build_pretrain_loader(cfg: TrainConfig, device: torch.device) -> DataLoader:
    """Shuffled loader of two-view batches. ``drop_last`` keeps the negative count constant."""
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        build_pretrain_dataset(cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", generator=generator,
        persistent_workers=False)


def build_eval_loader(dataset: Dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=num_workers, pin_memory=device.type == "cuda")
