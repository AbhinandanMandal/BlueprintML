"""Training step, training loop, checkpointing and resume."""

from __future__ import annotations

import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import MoCoConfig
from .models.encoder import MoCoEncoder
from .models.moco import MoCo
from .utils import peak_gpu_memory_mb, write_csv


def build_model(cfg: MoCoConfig, device: torch.device) -> MoCo:
    encoder = MoCoEncoder(cfg.backbone, cfg.embedding_dim, cfg.cifar_stem)
    model = MoCo(encoder, cfg.embedding_dim, cfg.queue_size, cfg.momentum,
                 cfg.temperature, cfg.bn_groups)
    return model.to(device)


def build_optimizer(cfg: MoCoConfig, model: MoCo) -> torch.optim.Optimizer:
    # Only the query encoder is optimised; the key encoder is updated by momentum.
    params = [p for p in model.encoder_q.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.sgd_momentum,
                           weight_decay=cfg.weight_decay)


def build_scheduler(cfg: MoCoConfig, optimizer: torch.optim.Optimizer):
    """Per-epoch schedule. 'step' mirrors the paper's x0.1 decays at 60%/80% of training."""
    import math
    milestones = sorted({int(f * cfg.epochs) for f in cfg.lr_milestones if int(f * cfg.epochs) > 0})

    def factor(epoch: int) -> float:
        if cfg.lr_schedule == "none":
            return 1.0
        if cfg.lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.epochs))
        return cfg.lr_gamma ** sum(epoch >= m for m in milestones)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_step(model: MoCo, optimizer: torch.optim.Optimizer, x_q: torch.Tensor,
               x_k: torch.Tensor, scaler: Optional[torch.amp.GradScaler] = None,
               amp: bool = False) -> Dict[str, float]:
    """One MoCo iteration.

    Order (momentum update first, as in the official forward(); enqueue after the loss):
      1 momentum update of f_k  2 q = f_q(x_q) (grad)  3 k = f_k(x_k) (no grad)
      4 logits/targets  5 CE loss  6 backward + SGD step on f_q  7 enqueue keys.
    The official code enqueues before the optimizer step; the queue is not part of the
    graph (the logits already used a cloned snapshot), so the result is identical.
    """
    model.train()
    model.momentum_update()                                    # step 3 of the spec
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x_q.device.type, enabled=amp):
        out = model(x_q, x_k)                                  # steps 4-9
    loss = model.loss(out)                                     # step 11
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()                          # step 12 (f_q only)
        scaler.step(optimizer)                                 # step 13
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    model.enqueue(out.k)                                       # step 14 (detached)

    with torch.no_grad():
        l_pos = out.logits[:, 0]
        acc1 = (out.logits.argmax(dim=1) == out.targets).float().mean().item()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss. Check lr, temperature and AMP settings.")
    return {"loss": float(loss.item()), "acc1": acc1,
            "pos_logit": float(l_pos.mean().item()),
            "queue_ptr": int(model.queue_ptr.item())}


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, model: MoCo, optimizer, scheduler, epoch: int,
                    global_step: int, cfg: MoCoConfig, history: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "encoder_q": model.encoder_q.state_dict(),
        "encoder_k": model.encoder_k.state_dict(),
        "queue": model.queue.detach().cpu(),
        "queue_ptr": model.queue_ptr.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,                  # number of COMPLETED epochs
        "global_step": global_step,
        "config": cfg.to_dict(),
        "seed": cfg.seed,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    Path(tmp).replace(path)  # atomic: an interrupted save never corrupts the last checkpoint


def load_checkpoint(path: str, model: MoCo, optimizer=None, scheduler=None,
                    restore_rng: bool = True, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint produced by ``save_checkpoint`` (a trusted, local file:
    it contains RNG states, so it is loaded with weights_only=False)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    missing = {"encoder_q", "encoder_k", "queue", "queue_ptr", "epoch", "global_step", "config"} - set(state)
    if missing:
        raise KeyError(f"Checkpoint {path} is missing keys: {sorted(missing)}")
    if tuple(state["queue"].shape) != tuple(model.queue.shape):
        raise ValueError(
            f"Checkpoint queue shape {tuple(state['queue'].shape)} != model queue shape "
            f"{tuple(model.queue.shape)}. Use the same queue_size/embedding_dim as the run that saved it.")
    model.encoder_q.load_state_dict(state["encoder_q"])
    model.encoder_k.load_state_dict(state["encoder_k"])
    with torch.no_grad():
        model.queue.copy_(state["queue"].to(model.queue.device))
        model.queue_ptr.copy_(state["queue_ptr"].to(model.queue_ptr.device))
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and state.get("rng"):
        r = state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"].cpu())
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
    return state


# ----------------------------------------------------------------------------- loop
@dataclass
class TrainResult:
    history: List[Dict[str, Any]] = field(default_factory=list)
    global_step: int = 0
    epochs_done: int = 0
    runtime_s: float = 0.0
    peak_gpu_mb: Optional[float] = None
    checkpoint_path: Optional[str] = None
    interrupted: bool = False


def train(cfg: MoCoConfig, model: MoCo, loader: DataLoader, device: torch.device,
          resume_from: Optional[str] = None, tag: str = "run") -> TrainResult:
    """Epoch-based training with an optional max_steps cap, periodic checkpoints,
    resume support, and a checkpoint on Ctrl-C / SIGTERM (Colab disconnects)."""
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    result = TrainResult()
    start_epoch = 0
    if resume_from is not None:
        state = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch, result.global_step = int(state["epoch"]), int(state["global_step"])
        result.history = list(state.get("history", []))
        print(f"Resumed from {resume_from}: epoch={start_epoch} step={result.global_step}")

    ckpt_path = str(Path(cfg.checkpoint_dir) / f"{tag}_latest.pt")
    result.checkpoint_path = ckpt_path
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _on_sigterm(signum, frame):  # Colab sends SIGTERM when a runtime is reclaimed
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # not in the main thread (e.g. some notebook kernels)
        pass

    t0 = time.time()
    epoch = start_epoch
    try:
        while epoch < cfg.epochs:
            if cfg.max_steps is not None and result.global_step >= cfg.max_steps:
                break
            epoch_losses: List[float] = []
            for x_q, x_k in loader:
                x_q = x_q.to(device, non_blocking=True)
                x_k = x_k.to(device, non_blocking=True)
                m = train_step(model, optimizer, x_q, x_k, scaler, use_amp)
                result.global_step += 1
                epoch_losses.append(m["loss"])
                row = {"step": result.global_step, "epoch": epoch + 1, "loss": m["loss"],
                       "acc1": m["acc1"], "pos_logit": m["pos_logit"],
                       "lr": optimizer.param_groups[0]["lr"], "queue_ptr": m["queue_ptr"],
                       "runtime_s": time.time() - t0,
                       "gpu_mb": peak_gpu_memory_mb(device) or float("nan")}
                result.history.append(row)
                if result.global_step % cfg.log_every == 0:
                    print(f"[ep {epoch+1}/{cfg.epochs} step {result.global_step}] "
                          f"loss {m['loss']:.4f} acc1 {m['acc1']:.3f} lr {row['lr']:.5f} "
                          f"ptr {m['queue_ptr']}")
                if cfg.max_steps is not None and result.global_step >= cfg.max_steps:
                    break
            scheduler.step()
            epoch += 1
            result.epochs_done = epoch
            if epoch % cfg.checkpoint_every_epochs == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch,
                                result.global_step, cfg, result.history)
    except KeyboardInterrupt:
        result.interrupted = True
        print("Interrupted: saving checkpoint before exit.")

    result.epochs_done = epoch
    save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, result.global_step,
                    cfg, result.history)
    result.runtime_s = time.time() - t0
    result.peak_gpu_mb = peak_gpu_memory_mb(device)
    write_csv(str(Path(cfg.output_dir) / f"{tag}_metrics.csv"), result.history)
    return result
