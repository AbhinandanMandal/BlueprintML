#!/usr/bin/env python
"""SimCLR pre-training. Run from the repository root, e.g.

    python scripts/train.py --output-dir ./outputs/smoke_test --epochs 3
    python scripts/train.py --config configs/research.yaml --output-dir ./outputs/research_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simclr.config import TrainConfig, add_config_arguments, build_config  # noqa: E402
from simclr.engine import fit  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="SimCLR pre-training (CIFAR-10, scaled down).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="Optional YAML file; CLI flags override it.")
    add_config_arguments(parser, TrainConfig)
    args = parser.parse_args(argv)
    cfg = build_config(TrainConfig, args, args.config)
    summary = fit(cfg)
    print("[done]", summary)


if __name__ == "__main__":
    main()
