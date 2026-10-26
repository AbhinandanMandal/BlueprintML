"""Classification metrics computed from a confusion matrix (rows = true, cols = predicted)."""
from __future__ import annotations

from typing import List, Optional

import torch


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds, targets = preds.long().flatten().cpu(), targets.long().flatten().cpu()
    if preds.shape != targets.shape:
        raise ValueError("preds and targets must have the same number of elements")
    if len(targets) and (targets.min() < 0 or targets.max() >= num_classes
                         or preds.min() < 0 or preds.max() >= num_classes):
        raise ValueError(f"labels/predictions must lie in [0, {num_classes})")
    idx = targets * num_classes + preds
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)


def top1_accuracy(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    return float(cm.diag().sum().item() / total) if total else 0.0


def per_class_accuracy(cm: torch.Tensor) -> List[Optional[float]]:
    """Recall per class; ``None`` for classes with no samples."""
    support = cm.sum(dim=1)
    return [float(cm[i, i].item() / support[i].item()) if support[i] > 0 else None
            for i in range(cm.shape[0])]


def macro_f1(cm: torch.Tensor) -> float:
    """Mean F1 over classes that appear in the labels or the predictions."""
    tp = cm.diag().double()
    fp = cm.sum(dim=0).double() - tp
    fn = cm.sum(dim=1).double() - tp
    denom = 2 * tp + fp + fn
    valid = denom > 0
    if not valid.any():
        return 0.0
    return float((2 * tp[valid] / denom[valid]).mean().item())
