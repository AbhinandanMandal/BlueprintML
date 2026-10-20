"""The MoCo v1 model: query encoder, momentum key encoder, FIFO queue."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import info_nce_loss, moco_logits


@dataclass
class MoCoOutput:
    logits: torch.Tensor   # [N, 1+K], already divided by tau
    targets: torch.Tensor  # [N], zeros (positive is column 0)
    q: torch.Tensor        # [N, C]
    k: torch.Tensor        # [N, C], detached


class MoCo(nn.Module):
    """Momentum Contrast (He et al., 2020).

    Registered buffers ``queue`` [K, C] and ``queue_ptr`` [1] are part of state_dict,
    so they travel with checkpoints and follow ``.to(device)``.

    Queue warm-up: like the official code, the queue starts as random unit vectors,
    so the first K/N steps contain "fake" negatives that are replaced FIFO by real
    keys. Documented in the README; no special masking is applied.

    Shuffling BN: the paper shuffles the key batch across GPUs so that a query and
    its positive key never share BN statistics. On a single device we emulate this
    with ``bn_groups`` > 1: the key batch is randomly permuted, encoded as
    ``bn_groups`` independent sub-batches (each with its own BN statistics), then
    un-permuted; the query batch is encoded as the same number of sub-batches in the
    original order. With ``bn_groups=1`` no emulation is done.
    """

    def __init__(self, encoder: nn.Module, embedding_dim: int, queue_size: int,
                 momentum: float, temperature: float, bn_groups: int = 1) -> None:
        super().__init__()
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if queue_size < 1 or bn_groups < 1:
            raise ValueError("queue_size and bn_groups must be >= 1")
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.bn_groups = bn_groups

        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)  # identical parameters at initialisation
        for p in self.encoder_k.parameters():
            p.requires_grad = False              # key encoder never gets gradients

        self.register_buffer("queue", F.normalize(torch.randn(queue_size, embedding_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def momentum_update(self) -> None:
        """theta_k <- m * theta_k + (1 - m) * theta_q   (paper Eq. 2).

        Only parameters are averaged. BN running statistics of the key encoder are
        updated by its own forward passes (as in the official implementation).
        """
        m = self.momentum
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.mul_(m).add_(p_q.detach(), alpha=1.0 - m)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """Insert a batch of keys at the pointer (FIFO), wrapping around if needed."""
        keys = keys.detach().to(device=self.queue.device, dtype=self.queue.dtype)
        n = keys.shape[0]
        k_size = self.queue_size
        if keys.shape[1] != self.embedding_dim:
            raise ValueError(f"keys have dim {keys.shape[1]}, expected {self.embedding_dim}")
        if n > k_size:
            raise ValueError(f"batch size {n} exceeds queue size {k_size}")
        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= k_size:
            self.queue[ptr:end] = keys
        else:  # wrap-around: fill to the end, then continue from index 0
            first = k_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end - k_size] = keys[first:]
        self.queue_ptr[0] = end % k_size

    # ------------------------------------------------------------------
    def _encode_split(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.bn_groups == 1:
            return encoder(x)
        return torch.cat([encoder(chunk) for chunk in x.chunk(self.bn_groups)], dim=0)

    @torch.no_grad()
    def _encode_keys(self, x_k: torch.Tensor) -> torch.Tensor:
        if self.bn_groups == 1:
            return self.encoder_k(x_k)
        perm = torch.randperm(x_k.shape[0], device=x_k.device)
        out = self._encode_split(self.encoder_k, x_k[perm])
        keys = torch.empty_like(out)
        keys[perm] = out  # undo the shuffle: out[j] belongs to original index perm[j]
        return keys

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> MoCoOutput:
        """Encode both views and build logits. Does NOT update momentum/queue/optimizer.

        The caller (training step) controls ordering; see train.train_step.
        """
        q = self._encode_split(self.encoder_q, x_q)   # gradients flow through f_q
        k = self._encode_keys(x_k).detach()           # stop-gradient on keys
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits, targets = moco_logits(
                q.float(), k.float(), self.queue.clone().detach(), self.temperature
            )
        return MoCoOutput(logits=logits, targets=targets, q=q, k=k)

    def loss(self, out: MoCoOutput) -> torch.Tensor:
        return info_nce_loss(out.logits, out.targets)
