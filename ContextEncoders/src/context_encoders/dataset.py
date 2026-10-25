"""CelebA image dataset, transforms and train/val loaders"""
import os
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def build_transform(image_size: int = 128) -> transforms.Compose:
    """Resize -> ToTensor -> Normalize to [-1, 1]. Same pipeline for train and validation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


class CelebADataset(Dataset):
    """Label-free dataset over a flat directory of images (e.g. ``img_align_celeba``)."""

    def __init__(self, img_dir, transform=None):
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found: '{img_dir}'. Pass the folder containing the CelebA "
                f"images (e.g. img_align_celeba) via --data-dir.")
        self.img_dir = img_dir
        self.img_names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
        if not self.img_names:
            raise FileNotFoundError(f"No {IMG_EXTS} images found in '{img_dir}'.")
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:  # corrupt / unreadable file
            raise RuntimeError(f"Could not read image '{img_path}': {e}") from e
        if self.transform:
            image = self.transform(image)
        return image


def split_indices(n: int, train_size: int, val_size: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Random disjoint train/val index split"""
    if train_size + val_size > n:
        raise ValueError(f"train_size + val_size = {train_size + val_size} exceeds dataset size {n}.")
    indices = np.random.default_rng(seed).permutation(n)
    return indices[:train_size], indices[train_size:train_size + val_size]


def make_loaders(data_dir, train_size=8000, val_size=1000, batch_size=32, num_workers=2,
                 image_size=128, split_seed=0, pin_memory=True):
    """Return ``(train_loader, val_loader, (train_idx, val_idx))``."""
    dataset = CelebADataset(img_dir=data_dir, transform=build_transform(image_size))
    train_idx, val_idx = split_indices(len(dataset), train_size, val_size, split_seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, (train_idx, val_idx)
