"""Base encoder f(.) and projection head g(.) (paper Sec. 2.1, 2.2).

h = f(x) is the representation used downstream; z = g(h) is only used by the
contrastive loss. They are separate modules so ``h`` can be extracted alone.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from .config import TrainConfig

FEATURE_DIMS = {"resnet18": 512, "resnet50": 2048}


def build_encoder(name: str = "resnet18", cifar_stem: bool = True) -> Tuple[nn.Module, int]:
    """Randomly initialised ResNet returning pooled features (no classifier).

    With ``cifar_stem`` the 7x7/stride-2 conv is replaced by a 3x3/stride-1 conv
    and the max-pool is removed (suitable for 32x32 inputs).
    """
    if name == "resnet18":
        net = resnet18(weights=None)
    elif name == "resnet50":
        net = resnet50(weights=None)
    else:
        raise ValueError(
            f"Unsupported encoder {name!r}; choose from {tuple(FEATURE_DIMS)}")
    feature_dim = net.fc.in_features
    assert feature_dim == FEATURE_DIMS[name]
    if cifar_stem:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(
            net.conv1.weight, mode="fan_out", nonlinearity="relu")
        net.maxpool = nn.Identity()
    net.fc = nn.Identity()  # remove the supervised classification head
    return net, feature_dim


class ProjectionHead(nn.Module):
    """z = W2 * ReLU(BN(W1 h)): one hidden layer, ReLU, optional hidden BatchNorm."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128, use_bn: bool = True) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, projector: ProjectionHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Representation h (what linear evaluation uses)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h, self.projector(h)


def build_model(cfg: TrainConfig) -> SimCLRModel:
    encoder, feature_dim = build_encoder(cfg.encoder, cfg.cifar_stem)
    projector = ProjectionHead(
        feature_dim, cfg.projection_hidden_dim, cfg.projection_dim, cfg.projection_bn)
    return SimCLRModel(encoder, feature_dim, projector)
