#!/usr/bin/env python
"""Quick kNN diagnostic of a checkpoint (no classifier training).

    python scripts/evaluate.py --checkpoint ./outputs/smoke_test/last_checkpoint.pt --data-dir ./data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simclr.checkpoint import load_model_for_eval  # noqa: E402
from simclr.datasets import build_eval_loader, build_labeled_dataset  # noqa: E402
from simclr.engine import knn_monitor  # noqa: E402
from simclr.utils import resolve_device  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="kNN evaluation of a SimCLR checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--bank-size", type=int, default=0, help="Train images in the kNN bank (0 = all).")
    p.add_argument("--test-size", type=int, default=0, help="Test images (0 = full test split).")
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, cfg = load_model_for_eval(args.checkpoint, device)
    bank = build_labeled_dataset(cfg.dataset, args.data_dir, True, args.bank_size, args.subset_seed, cfg.image_size)
    query = build_labeled_dataset(cfg.dataset, args.data_dir, False, args.test_size, args.subset_seed + 1,
                                  cfg.image_size)
    acc = knn_monitor(model, build_eval_loader(bank, 512, args.num_workers, device),
                      build_eval_loader(query, 512, args.num_workers, device), device)
    print(f"kNN top-1 (k=20): {acc:.4f} | bank={len(bank)} | test={len(query)} | seed={args.subset_seed}")


if __name__ == "__main__":
    main()
