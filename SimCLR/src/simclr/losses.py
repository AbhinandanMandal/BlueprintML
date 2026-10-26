"""NT-Xent loss (paper Eq. 1).

For N images there are 2N projections. For a positive pair (i, j):
    l(i, j) = -log( exp(sim(z_i, z_j)/tau) / sum_{k != i} exp(sim(z_i, z_k)/tau) )
and the loss is the mean over all 2N anchors, i.e. both directions of every pair.
The positive stays in the denominator; only the self-similarity is masked.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> None:
    if z_i.ndim != 2 or z_j.ndim != 2:
        raise ValueError(
            f"Expected 2-D [N, D] embeddings, got {tuple(z_i.shape)} and {tuple(z_j.shape)}")
    if z_i.shape != z_j.shape:
        raise ValueError(
            f"The two views must have equal shape, got {tuple(z_i.shape)} vs {tuple(z_j.shape)}")
    if z_i.shape[0] < 2:
        raise ValueError(
            "NT-Xent needs a batch of at least 2 images to have negatives")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")


def positive_indices(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Index of each anchor's positive in the stacked [z_i; z_j] batch of size 2N."""
    return (torch.arange(2 * n, device=device) + n) % (2 * n)


def similarity_logits(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled cosine similarities, shape [2N, 2N], diagonal masked to the dtype minimum."""
    z = F.normalize(torch.cat([z_i, z_j], dim=0).float(), dim=1)
    logits = z @ z.t() / temperature
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    return logits.masked_fill(eye, torch.finfo(logits.dtype).min)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Vectorised NT-Xent. Computed in float32 even under mixed precision."""
    _validate(z_i, z_j, temperature)
    logits = similarity_logits(z_i, z_j, temperature)
    targets = positive_indices(z_i.shape[0], logits.device)
    return F.cross_entropy(logits, targets)  # mean over 2N anchors


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return nt_xent_loss(z_i, z_j, self.temperature)
