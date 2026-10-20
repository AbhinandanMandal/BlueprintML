"""Datasets and augmentations.

The self-supervised dataset returns ONLY two augmented views of an image; labels are
dropped inside the dataset so they cannot reach the contrastive loss. Labelled data is
exposed separately and used only by ``evaluate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import MoCoConfig
from .utils import worker_init_fn

# CIFAR-10 channel statistics (the official ImageNet recipe uses ImageNet statistics;
# this is a CIFAR adaptation).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def moco_v1_transform(image_size: int) -> transforms.Compose:
    """MoCo v1 augmentation (official main_moco.py, non-aug-plus branch).

    RandomResizedCrop(scale=(0.2, 1)), RandomGrayscale(p=0.2), ColorJitter(0.4, 0.4, 0.4, 0.4),
    RandomHorizontalFlip, ToTensor, Normalize. The crop size is ``image_size`` (32 for
    CIFAR) instead of 224: a CIFAR-specific adaptation. No Gaussian blur (that is v2).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for feature extraction (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


class TwoCropsTransform:
    """Apply the same stochastic policy twice with independent random draws."""

    def __init__(self, base: Callable) -> None:
        self.base = base

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        return [self.base(img), self.base(img)]


class SyntheticImages(Dataset):
    """Tiny in-memory stand-in (10 noisy colour/stripe classes) for checking MECHANICS only.

    Used when CIFAR-10 cannot be downloaded (e.g. offline sandbox). Results on it must
    never be reported as CIFAR-10 / research results.
    """

    def __init__(self, n: int, seed: int, size: int = 32) -> None:
        rng = np.random.RandomState(seed)
        self.targets = rng.randint(0, 10, size=n).tolist()
        yy, xx = np.mgrid[0:size, 0:size] / size
        imgs = np.empty((n, size, size, 3), dtype=np.uint8)
        for i, c in enumerate(self.targets):
            hue = np.array([(c * 37) % 255, (c * 91) % 255, (c * 173) % 255], dtype=np.float32)
            angle = c * np.pi / 10 + rng.uniform(-0.6, 0.6)
            stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (3 + c % 3) * (xx * np.cos(angle) + yy * np.sin(angle)))
            img = hue[None, None, :] * (0.4 + 0.6 * stripes[..., None]) + rng.normal(0, 70, (size, size, 3))
            imgs[i] = np.clip(img, 0, 255).astype(np.uint8)
        self.data = imgs

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        return Image.fromarray(self.data[idx]), self.targets[idx]


class SSLDataset(Dataset):
    """Wraps a labelled dataset; returns (view_q, view_k) and DISCARDS the label."""

    def __init__(self, base: Dataset, indices: Sequence[int], transform: TwoCropsTransform) -> None:
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, _label = self.base[self.indices[i]]  # label intentionally unused
        x_q, x_k = self.transform(img)
        return x_q, x_k


class LabelledDataset(Dataset):
    """(image tensor, label) pairs with a deterministic transform; evaluation only."""

    def __init__(self, base: Dataset, indices: Sequence[int], transform: Callable) -> None:
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[self.indices[i]]
        return self.transform(img), int(label)


@dataclass
class DataBundle:
    ssl_dataset: SSLDataset
    eval_train: LabelledDataset
    eval_test: LabelledDataset
    ssl_indices: List[int]
    source: str  # "cifar10" or "synthetic"


def _fixed_subset(n_total: int, n: int, seed: int) -> List[int]:
    """Reproducible subset: the first n of a seed-determined permutation."""
    if n > n_total:
        raise ValueError(f"Requested subset of {n} images but only {n_total} are available")
    return np.random.RandomState(seed).permutation(n_total)[:n].tolist()


def build_data(cfg: MoCoConfig) -> DataBundle:
    """Build SSL / eval datasets. Subset indices depend only on ``cfg.seed``.

    * SSL training uses a subset of the CIFAR-10 TRAIN split (labels discarded).
    * The linear probe / kNN reference set is a labelled subset of the TRAIN split.
    * Reported accuracy is on a subset of the TEST split, which never enters SSL
      training or hyper-parameter selection.
    """
    if cfg.dataset == "cifar10":
        train_base = datasets.CIFAR10(cfg.data_dir, train=True, download=True)
        test_base = datasets.CIFAR10(cfg.data_dir, train=False, download=True)
        source = "cifar10"
    else:
        n_train = max(cfg.subset_size, cfg.eval_train_size)
        train_base = SyntheticImages(n_train, cfg.seed, cfg.image_size)
        test_base = SyntheticImages(min(cfg.eval_test_size, 2000), cfg.seed + 1, cfg.image_size)
        source = "synthetic"

    ssl_idx = _fixed_subset(len(train_base), cfg.subset_size, cfg.seed)
    eval_train_idx = _fixed_subset(len(train_base), cfg.eval_train_size, cfg.seed + 1)
    eval_test_idx = _fixed_subset(len(test_base), min(cfg.eval_test_size, len(test_base)), cfg.seed + 2)

    return DataBundle(
        ssl_dataset=SSLDataset(train_base, ssl_idx, TwoCropsTransform(moco_v1_transform(cfg.image_size))),
        eval_train=LabelledDataset(train_base, eval_train_idx, eval_transform(cfg.image_size)),
        eval_test=LabelledDataset(test_base, eval_test_idx, eval_transform(cfg.image_size)),
        ssl_indices=ssl_idx,
        source=source,
    )


def make_ssl_loader(dataset: Dataset, cfg: MoCoConfig) -> DataLoader:
    """Shuffled loader that drops the last partial batch (BN needs >1 sample; queue
    insertion assumes a fixed batch size)."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, worker_init_fn=worker_init_fn, generator=g,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def make_eval_loader(dataset: Dataset, batch_size: int = 256, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
