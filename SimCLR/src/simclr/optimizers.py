"""LARS, SGD factory and the warm-up + cosine schedule used by SimCLR."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch
from torch.optim import Optimizer

from .config import TrainConfig


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling with momentum (You et al. 2017).

    Follows the structure of the official SimCLR implementation: for adapted
    parameters, g = grad + wd * w and the step is scaled by
    ``eta * ||w|| / ||g||``. Groups with ``lars_exclude=True`` (biases / BN)
    get neither weight decay nor trust-ratio scaling.
    """

    def __init__(self, params: Iterable, lr: float, momentum: float = 0.9, weight_decay: float = 0.0,
                 trust_coefficient: float = 0.001, lars_exclude: bool = False) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum and weight_decay must be non-negative")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, lars_exclude=lars_exclude)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not group["lars_exclude"]:
                    if group["weight_decay"] != 0:
                        g = g.add(p, alpha=group["weight_decay"])
                    w_norm, g_norm = torch.norm(p), torch.norm(g)
                    trust = torch.where((w_norm > 0) & (g_norm > 0),
                                        group["trust_coefficient"] * w_norm / g_norm,
                                        torch.ones_like(w_norm))
                    g = g * trust
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g, alpha=group["lr"])
                p.sub_(buf)
        return loss


def _param_groups(model: torch.nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)  # biases and BN affine params
    return [{"params": decay, "weight_decay": weight_decay, "lars_exclude": False},
            {"params": no_decay, "weight_decay": 0.0, "lars_exclude": True}]


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> Optimizer:
    groups = _param_groups(model, cfg.wd)
    if cfg.optimizer == "lars":
        return LARS(groups, lr=cfg.lr, momentum=cfg.momentum, trust_coefficient=cfg.lars_trust_coefficient)
    for g in groups:
        g.pop("lars_exclude")
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)


class WarmupCosineScheduler:
    """Per-iteration linear warm-up followed by cosine decay to ``min_lr``.

    ``lr(s)`` is a pure function of the step ``s``: ``base*(s+1)/warmup`` during
    warm-up, then ``min + 0.5*(base-min)*(1+cos(pi*progress))``. Call ``step()``
    once after every optimizer step. State is just the step counter, so the
    schedule can be rebuilt with a longer horizon when extending a run.
    """

    def __init__(self, optimizer: Optimizer, base_lr: float, total_steps: int,
                 warmup_steps: int = 0, min_lr: float = 0.0) -> None:
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("need total_steps >= 1 and 0 <= warmup_steps < total_steps")
        self.optimizer, self.base_lr, self.total_steps = optimizer, base_lr, total_steps
        self.warmup_steps, self.min_lr = warmup_steps, min_lr
        self.step_count = 0
        self._apply()

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _apply(self) -> None:
        self.last_lr = self.lr_at(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def state_dict(self) -> Dict[str, Any]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self._apply()
