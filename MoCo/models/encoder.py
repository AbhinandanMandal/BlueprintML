"""ResNet encoder: randomly initialised backbone + linear projection + L2 norm."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MoCoEncoder(nn.Module):
    """torchvision ResNet (no pretrained weights) whose fc is replaced by a Linear
    layer mapping pooled features to ``embedding_dim``; output is L2-normalised.

    This matches MoCo v1: the *last fully-connected layer* gives the 128-D output
    (no MLP head -- that is a MoCo v2 change).

    CIFAR adaptation (``cifar_stem=True``): conv1 becomes a 3x3 stride-1 conv and the
    max-pool is removed, because the standard 7x7/stride-2 stem plus max-pool would
    shrink 32x32 inputs to 8x8 before the first residual stage. This is NOT part of
    the paper (which uses the unmodified ResNet-50 on 224x224 ImageNet crops).
    """

    def __init__(self, backbone: str = "resnet18", embedding_dim: int = 128,
                 cifar_stem: bool = True) -> None:
        super().__init__()
        if backbone not in ("resnet18", "resnet50"):
            raise ValueError(f"Unsupported backbone {backbone!r}; use resnet18 or resnet50")
        net = getattr(torchvision.models, backbone)(weights=None)  # random init only
        self.feature_dim: int = net.fc.in_features
        if cifar_stem:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Identity()  # backbone now returns pooled features [N, feature_dim]
        self.backbone = net
        self.fc = nn.Linear(self.feature_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features (what the linear probe / kNN evaluates)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised embedding [N, embedding_dim] (computed in fp32)."""
        emb = self.fc(self.features(x)).float()
        return F.normalize(emb, dim=1)
