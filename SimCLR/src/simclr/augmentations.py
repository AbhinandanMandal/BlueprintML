"""SimCLR data augmentation (paper Sec. 3, Appendix A).

Each image is transformed twice by the *same stochastic transform*; independent
random draws on each call give two different views t(x) and t'(x).
Pipeline: random resized crop -> horizontal flip -> colour jitter (p=0.8,
(0.8s, 0.8s, 0.8s, 0.2s)) -> random grayscale (p=0.2) -> optional Gaussian blur.
Blur is off by default because the paper does not use it on CIFAR-10.
The module never sees labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
NATIVE_SIZE = 32


def _odd_kernel_size(image_size: int, fraction: float = 0.1) -> int:
    """Blur kernel = ~10% of image side (paper), forced odd and >= 3."""
    k = max(3, int(round(image_size * fraction)))
    return k if k % 2 == 1 else k + 1


def build_train_transform(
    image_size: int = NATIVE_SIZE,
    crop_scale: Tuple[float, float] = (0.08, 1.0),
    color_strength: float = 0.5,
    blur: bool = False,
    blur_prob: float = 0.5,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> T.Compose:
    """Stochastic training transform: PIL image -> normalised tensor [3, S, S]."""
    s = color_strength
    steps = [
        T.RandomResizedCrop(image_size, scale=crop_scale,
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]
    if blur:
        steps.append(T.RandomApply(
            [T.GaussianBlur(kernel_size=_odd_kernel_size(image_size), sigma=(0.1, 2.0))], p=blur_prob))
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


def build_eval_transform(
    image_size: int = NATIVE_SIZE,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> T.Compose:
    """Deterministic evaluation transform (resize only when upsampling)."""
    steps = []
    if image_size != NATIVE_SIZE:
        steps.append(
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC))
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


class TwoViewTransform:
    """Apply one stochastic transform twice -> two independently augmented views."""

    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_transform(image), self.base_transform(image)


def denormalize(x: torch.Tensor, mean: Sequence[float] = CIFAR10_MEAN,
                std: Sequence[float] = CIFAR10_STD) -> torch.Tensor:
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * s + m).clamp(0, 1)


def visualize_pairs(dataset, path: str, num_images: int = 8,
                    mean: Sequence[float] = CIFAR10_MEAN, std: Sequence[float] = CIFAR10_STD) -> Path:
    """Save a grid with one row per source image and the two views as columns.

    ``dataset`` must yield ``(view1, view2)`` pairs (see ``datasets.SimCLRDataset``).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "matplotlib is required for visualize_pairs (pip install matplotlib)") from exc

    n = min(num_images, len(dataset))
    fig, axes = plt.subplots(n, 2, figsize=(3, 1.6 * n), squeeze=False)
    for row in range(n):
        v1, v2 = dataset[row]
        for col, view in enumerate((v1, v2)):
            axes[row][col].imshow(denormalize(
                view, mean, std).permute(1, 2, 0).numpy())
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(f"view {col + 1}", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
