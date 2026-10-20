"""Lightweight downstream evaluation: kNN and linear probe on frozen backbone features.

Labels are used ONLY here. These are diagnostics on CIFAR-10, not a reproduction of the
paper's ImageNet linear-evaluation protocol (lr 30, wd 0, 100 epochs on ResNet-50).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models.encoder import MoCoEncoder


@torch.no_grad()
def extract_features(encoder: MoCoEncoder, loader: DataLoader,
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen pooled backbone features (before the projection fc) and labels."""
    was_training = encoder.training
    encoder.eval()  # BN uses running statistics; no augmentation
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder.features(x.to(device)).float().cpu())
        labels.append(y)
    encoder.train(was_training)
    return torch.cat(feats), torch.cat(labels)


def knn_top1(train_f: torch.Tensor, train_y: torch.Tensor,
             test_f: torch.Tensor, test_y: torch.Tensor) -> float:
    """Top-1 nearest-neighbour accuracy by cosine similarity (L2-normalised features)."""
    a, b = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    nearest = (b @ a.t()).argmax(dim=1)
    return (train_y[nearest] == test_y).float().mean().item()


def knn_vote(train_f, train_y, test_f, test_y, k: int = 20, num_classes: int = 10) -> float:
    """Majority-vote k-NN accuracy (cosine similarity)."""
    a, b = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    k = min(k, a.shape[0])
    idx = (b @ a.t()).topk(k, dim=1).indices
    votes = F.one_hot(train_y[idx], num_classes).sum(dim=1)
    return (votes.argmax(dim=1) == test_y).float().mean().item()


def linear_probe(train_f: torch.Tensor, train_y: torch.Tensor, test_f: torch.Tensor,
                 test_y: torch.Tensor, epochs: int, lr: float, weight_decay: float,
                 seed: int, num_classes: int = 10) -> float:
    """Train a linear classifier on frozen, standardised features (mean/std from the
    labelled TRAIN subset only) with Adam; report test accuracy once at the end."""
    g = torch.Generator().manual_seed(seed)
    mean, std = train_f.mean(0, keepdim=True), train_f.std(0, keepdim=True) + 1e-6
    xtr, xte = (train_f - mean) / std, (test_f - mean) / std
    clf = nn.Linear(xtr.shape[1], num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    n, bs = xtr.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(clf(xtr[b]), train_y[b]).backward()
            opt.step()
    with torch.no_grad():
        return (clf(xte).argmax(1) == test_y).float().mean().item()


def evaluate_representation(encoder: MoCoEncoder, eval_train, eval_test, device: torch.device,
                            epochs: int, lr: float, weight_decay: float, k: int,
                            seed: int) -> Dict[str, float]:
    tr = DataLoader(eval_train, batch_size=256, shuffle=False)
    te = DataLoader(eval_test, batch_size=256, shuffle=False)
    ftr, ytr = extract_features(encoder, tr, device)
    fte, yte = extract_features(encoder, te, device)
    return {
        "n_labelled_train": int(ftr.shape[0]),
        "n_test": int(fte.shape[0]),
        "knn_top1": knn_top1(ftr, ytr, fte, yte),
        f"knn_vote_k{k}": knn_vote(ftr, ytr, fte, yte, k),
        "linear_probe": linear_probe(ftr, ytr, fte, yte, epochs, lr, weight_decay, seed),
        "chance": 1.0 / 10,
    }
