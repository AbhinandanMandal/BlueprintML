#!/usr/bin/env python
"""Save a grid of paired augmented views for visual inspection.

    python scripts/visualize_augmentations.py --output ./outputs/aug_pairs.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simclr.augmentations import visualize_pairs  # noqa: E402
from simclr.config import TrainConfig  # noqa: E402
from simclr.datasets import build_pretrain_dataset  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Visualise paired SimCLR views.")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "synthetic"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="./outputs/aug_pairs.png")
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--color-strength", type=float, default=0.5)
    p.add_argument("--blur", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    cfg = TrainConfig(dataset=args.dataset, data_dir=args.data_dir, train_subset_size=max(args.num_images, 2),
                      batch_size=2, color_strength=args.color_strength, blur=args.blur, seed=args.seed)
    print("saved", visualize_pairs(build_pretrain_dataset(cfg), args.output, args.num_images))


if __name__ == "__main__":
    main()
