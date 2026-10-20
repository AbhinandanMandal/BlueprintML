"""InfoNCE logits and loss (paper Eq. 1) in the form of Algorithm 1."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def moco_logits(
    q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the (1+K)-way logits and targets.

    Args:
        q: [N, C] L2-normalised queries (gradients flow here).
        k: [N, C] L2-normalised positive keys (no gradient).
        queue: [K, C] L2-normalised negative keys (no gradient).
        temperature: tau.

    Returns:
        logits [N, 1+K] = cat([q.k+, q.k_i^-]) / tau, with the positive in column 0,
        and targets [N] of zeros (the index of the positive).
    """
    if q.shape != k.shape:
        raise ValueError(f"query/key shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")
    if queue.dim() != 2 or queue.shape[1] != q.shape[1]:
        raise ValueError(
            f"queue must be [K, {q.shape[1]}], got {tuple(queue.shape)}"
        )
    k = k.detach()
    queue = queue.detach()
    l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)   # [N, 1]  q . k+
    l_neg = q @ queue.t()                                   # [N, K]  q . k_i^-
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature  # [N, 1+K]
    targets = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    return logits, targets


def info_nce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the (1+K) logits == -log softmax prob of the positive key."""
    return F.cross_entropy(logits, targets)
