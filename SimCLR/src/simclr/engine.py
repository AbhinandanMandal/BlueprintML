"""Training engine: one epoch of SimCLR optimisation and the full fit/resume loop."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .checkpoint import build_payload, check_resume_compatibility, load_checkpoint, save_checkpoint
from .config import TrainConfig
from .datasets import build_eval_loader, build_labeled_dataset, build_pretrain_loader
from .evaluation import extract_features, knn_accuracy
from .losses import nt_xent_loss
from .models import SimCLRModel, build_model
from .optimizers import WarmupCosineScheduler, build_optimizer
from .utils import (CSVLogger, get_environment_info, resolve_device, seed_everything, set_rng_state,
                    write_json)

CSV_FIELDS = ["epoch", "global_step", "train_loss", "lr", "epoch_time_s", "images_per_sec",
              "peak_gpu_mem_mb", "knn_acc"]


def train_one_epoch(model: SimCLRModel, loader, optimizer, scheduler: WarmupCosineScheduler, scaler,
                    cfg: TrainConfig, device: torch.device, use_amp: bool) -> Dict[str, Any]:
    """One pass over the unlabelled two-view loader. Labels never appear in this loop."""
    model.train()
    loss_sum = torch.zeros((), device=device)  # accumulated on-device: no per-step host sync
    steps = images = 0
    lr_used = scheduler.last_lr
    start = time.perf_counter()
    for view_1, view_2 in loader:
        n = view_1.shape[0]
        x = torch.cat([view_1, view_2], dim=0).to(device, non_blocking=True)  # 2N views
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, z = model(x)  # h = f(x) inside, z = g(h)
        loss = nt_xent_loss(z[:n], z[n:], cfg.temperature)  # computed in fp32
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_used = scheduler.last_lr  # lr that this step actually used
        scheduler.step()
        loss_sum += loss.detach()
        steps += 1
        images += n
    mean_loss = (loss_sum / max(steps, 1)).item()
    if not torch.isfinite(torch.tensor(mean_loss)):
        raise FloatingPointError("Training loss is not finite. Try a lower --base-lr, enable --grad-clip, "
                                 "or disable --amp.")
    elapsed = time.perf_counter() - start
    return {"loss": mean_loss, "lr": lr_used, "steps": steps, "images": images,
            "time_s": elapsed, "images_per_sec": images / max(elapsed, 1e-9)}


def knn_monitor(model: SimCLRModel, bank_loader, query_loader, device: torch.device) -> float:
    """kNN accuracy of the frozen representation. Uses labels for evaluation only; no gradients."""
    bank_x, bank_y = extract_features(model, bank_loader, device)
    query_x, query_y = extract_features(model, query_loader, device)
    return knn_accuracy(bank_x, bank_y, query_x, query_y, device=device)


def _is_better(new: float, best: Optional[float], policy: str) -> bool:
    if best is None:
        return True
    return new > best if policy == "knn_acc" else new < best


def fit(cfg: TrainConfig) -> Dict[str, Any]:
    """Run (or resume) SimCLR pre-training and return a summary dict."""
    cfg.validate()
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed, cfg.deterministic)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    use_amp = cfg.amp and device.type == "cuda"
    if cfg.amp and not use_amp:
        print("[info] Mixed precision requested but running on CPU: AMP disabled.")

    model = build_model(cfg).to(device)
    loader = build_pretrain_loader(cfg, device)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise ValueError("Zero batches per epoch: reduce --batch-size or raise --train-subset-size.")
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(optimizer, cfg.lr, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor = None
    if cfg.eval_every > 0:
        bank = build_labeled_dataset(cfg.dataset, cfg.data_dir, True, cfg.train_subset_size,
                                     cfg.effective_subset_seed, cfg.image_size, cfg.download)
        query = build_labeled_dataset(cfg.dataset, cfg.data_dir, False, cfg.eval_subset_size,
                                      cfg.effective_subset_seed + 1, cfg.image_size, cfg.download)
        monitor = (build_eval_loader(bank, 512, cfg.num_workers, device),
                   build_eval_loader(query, 512, cfg.num_workers, device))

    policy = "knn_acc" if cfg.eval_every > 0 else "train_loss"
    start_epoch, global_step, best_metric = 0, 0, None

    resume_path = cfg.resume
    if resume_path == "auto":
        candidate = out / "last_checkpoint.pt"
        resume_path = str(candidate) if candidate.is_file() else None
        print(f"[resume] auto: {'found ' + str(candidate) if resume_path else 'no checkpoint, starting fresh'}")
    if resume_path:
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        check_resume_compatibility(ckpt["config"], cfg, cfg.extend_schedule)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])  # only the step counter; horizon comes from cfg
        if ckpt.get("scaler") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("rng_state") is not None:
            set_rng_state(ckpt["rng_state"])
        start_epoch, global_step = int(ckpt["epoch"]), int(ckpt["global_step"])
        if ckpt.get("best_metric_name") == policy:
            best_metric = ckpt.get("best_metric")
        print(f"[resume] {resume_path}: continuing from epoch {start_epoch} (global step {global_step})")
        if start_epoch >= cfg.epochs:
            print(f"[resume] checkpoint already completed {start_epoch} >= {cfg.epochs} epochs; nothing to do. "
                  "Use --extend-schedule with a larger --epochs to continue training.")

    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "environment.json", get_environment_info())
    logger = CSVLogger(out / "metrics.csv", CSV_FIELDS)

    print(f"[setup] device={device} amp={use_amp} encoder={cfg.encoder} feature_dim={model.feature_dim} "
          f"images={len(loader.dataset)} steps/epoch={steps_per_epoch} negatives/anchor={2 * cfg.batch_size - 2} "
          f"lr={cfg.lr:.4g} optimizer={cfg.optimizer}")

    last_loss: Optional[float] = None
    for epoch in range(start_epoch, cfg.epochs):
        loader.generator.manual_seed(cfg.seed + epoch)  # epoch-level determinism, independent of resume
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stats = train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, use_amp)
        global_step += stats["steps"]
        last_loss = stats["loss"]

        knn = None
        if monitor is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs):
            knn = knn_monitor(model, monitor[0], monitor[1], device)
            model.train()
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None

        logger.log({"epoch": epoch + 1, "global_step": global_step, "train_loss": round(stats["loss"], 6),
                    "lr": stats["lr"], "epoch_time_s": round(stats["time_s"], 3),
                    "images_per_sec": round(stats["images_per_sec"], 1),
                    "peak_gpu_mem_mb": None if peak is None else round(peak, 1),
                    "knn_acc": None if knn is None else round(knn, 4)})
        print(f"epoch {epoch + 1:>3}/{cfg.epochs} | loss {stats['loss']:.4f} | lr {stats['lr']:.4g} | "
              f"{stats['images_per_sec']:.0f} img/s | {stats['time_s']:.1f}s" +
              (f" | kNN {knn:.4f}" if knn is not None else ""))

        metric = knn if policy == "knn_acc" else stats["loss"]
        improved = metric is not None and _is_better(metric, best_metric, policy)
        if improved:
            best_metric = metric
        is_last = epoch + 1 == cfg.epochs
        if improved or (epoch + 1) % cfg.save_every == 0 or is_last:
            payload = build_payload(model, optimizer, scheduler, scaler, cfg, epoch + 1, global_step,
                                    best_metric, policy)
            if (epoch + 1) % cfg.save_every == 0 or is_last:
                save_checkpoint(out / "last_checkpoint.pt", payload)
            if improved:
                save_checkpoint(out / "best_checkpoint.pt", payload)

    summary = {"epochs_completed": cfg.epochs if last_loss is not None else start_epoch,
               "global_step": global_step, "final_train_loss": last_loss, "best_metric_name": policy,
               "best_metric": best_metric, "output_dir": str(out)}
    write_json(out / "summary.json", summary)
    return summary
