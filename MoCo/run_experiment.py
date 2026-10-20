"""CLI: python -m moco_reproduction.run_experiment --mode smoke|small [--dataset synthetic] ...

Runs: env report -> data -> train -> checkpoint reload check -> evaluation (+ random-init
baseline) -> plots. Never launches a full-dataset job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import MoCoConfig, small_config, smoke_config
from .datasets import build_data, make_ssl_loader
from .evaluate import evaluate_representation
from .plots import plot_training
from .train import build_model, build_optimizer, load_checkpoint, train
from .utils import environment_report, get_device, print_config, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "small"], default="smoke")
    ap.add_argument("--dataset", default=None, help="cifar10 | synthetic (mechanics only)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--linear-epochs", type=int, default=30)
    args = ap.parse_args()

    over = dict(output_dir=args.out, checkpoint_dir=f"{args.out}/checkpoints",
                data_dir=args.data_dir, num_workers=args.num_workers, device=args.device,
                linear_epochs=args.linear_epochs)
    if args.dataset:
        over["dataset"] = args.dataset
    if args.epochs:
        over["epochs"] = args.epochs
    cfg = (smoke_config if args.mode == "smoke" else small_config)(**over)

    print("Environment:", json.dumps(environment_report(cfg), indent=2))
    print_config(cfg)
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    data = build_data(cfg)
    print(f"data source={data.source} ssl_images={len(data.ssl_dataset)} "
          f"eval_train={len(data.eval_train)} eval_test={len(data.eval_test)}")
    loader = make_ssl_loader(data.ssl_dataset, cfg)
    model = build_model(cfg, device)
    res = train(cfg, model, loader, device, tag=args.mode)
    losses = [r["loss"] for r in res.history]
    print(f"\nTRAIN: steps={res.global_step} first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f} "
          f"mean_loss={sum(losses)/len(losses):.4f} queue_ptr={int(model.queue_ptr)} "
          f"lr={res.history[-1]['lr']:.5f} runtime={res.runtime_s:.1f}s peak_gpu_mb={res.peak_gpu_mb}")

    # checkpoint reload check
    reloaded = build_model(cfg, device)
    load_checkpoint(res.checkpoint_path, reloaded, build_optimizer(cfg, reloaded), map_location=device)
    same = all(torch.equal(a, b) for a, b in zip(model.encoder_q.parameters(), reloaded.encoder_q.parameters()))
    same &= torch.equal(model.queue, reloaded.queue) and int(model.queue_ptr) == int(reloaded.queue_ptr)
    print("checkpoint reload exact match:", same)

    # evaluation: trained encoder vs random-init encoder (same seed) as a reference
    kw = dict(epochs=cfg.linear_epochs, lr=cfg.linear_lr, weight_decay=cfg.linear_weight_decay,
              k=cfg.knn_k, seed=cfg.seed)
    trained = evaluate_representation(model.encoder_q, data.eval_train, data.eval_test, device, **kw)
    seed_everything(cfg.seed)
    random_init = evaluate_representation(build_model(cfg, device).encoder_q, data.eval_train,
                                          data.eval_test, device, **kw)
    results = {"mode": args.mode, "data_source": data.source, "trained": trained,
               "random_init_baseline": random_init, "config": cfg.to_dict(),
               "steps": res.global_step, "runtime_s": res.runtime_s}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.mode}_results.json").write_text(json.dumps(results, indent=2))
    print("EVAL trained    :", trained)
    print("EVAL random init:", random_init)
    plot_training(res.history, args.out, args.mode)


if __name__ == "__main__":
    main()
