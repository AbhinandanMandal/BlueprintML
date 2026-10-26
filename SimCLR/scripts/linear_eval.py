#!/usr/bin/env python
"""Linear evaluation of a frozen SimCLR encoder.

    python scripts/linear_eval.py --checkpoint ./outputs/smoke_test/last_checkpoint.pt \
        --data-dir ./data --output-dir ./outputs/linear_eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simclr.checkpoint import load_model_for_eval  # noqa: E402
from simclr.config import LinearEvalConfig, add_config_arguments, build_config  # noqa: E402
from simclr.evaluation import run_linear_evaluation  # noqa: E402
from simclr.utils import get_environment_info, resolve_device, write_json  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Frozen-encoder linear evaluation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, help="Optional YAML file; CLI flags override it.")
    add_config_arguments(parser, LinearEvalConfig)
    args = parser.parse_args(argv)
    cfg = build_config(LinearEvalConfig, args, args.config)

    device = resolve_device(cfg.device)
    model, train_cfg = load_model_for_eval(cfg.checkpoint, device)  # projection head is loaded but unused
    results = run_linear_evaluation(model, cfg, train_cfg, device)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["linear_eval_config"] = cfg.to_dict()
    results["environment"] = get_environment_info()
    write_json(out / "linear_eval_results.json", results)
    with (out / "confusion_matrix.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(results["test_confusion_matrix"])
    print(f"test top-1: {results['test_top1_accuracy']:.4f} | macro-F1: {results['test_macro_f1']:.4f} | "
          f"test loss: {results['test_loss']:.4f}")
    print(f"feature dim: {results['encoder_feature_dim']} | probe fit samples: "
          f"{results['num_labeled_train_samples_used_to_fit']} | test samples: {results['num_test_samples']}")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
