"""Feature extraction, kNN monitor and frozen-encoder linear evaluation (paper Sec. 6 / App. B.6).

Labels are used here ONLY for downstream evaluation, never in the pre-training loss.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LinearEvalConfig, TrainConfig
from .datasets import NUM_CLASSES, build_eval_loader, build_labeled_dataset
from .metrics import confusion_matrix, macro_f1, per_class_accuracy, top1_accuracy
from .models import SimCLRModel
from .utils import seed_everything


@torch.no_grad()
def extract_features(model: SimCLRModel, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen-encoder representations h (not z) and labels, returned on CPU in float32."""
    was_training = model.training
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model.encode(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y.cpu())
    model.train(was_training)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(bank_x: torch.Tensor, bank_y: torch.Tensor, query_x: torch.Tensor, query_y: torch.Tensor,
                 num_classes: int = NUM_CLASSES, k: int = 20, temperature: float = 0.1,
                 device: Optional[torch.device] = None, chunk: int = 1024) -> float:
    """Weighted cosine kNN accuracy (cheap representation-quality monitor)."""
    device = device or torch.device("cpu")
    bank = F.normalize(bank_x.to(device).float(), dim=1)
    query = F.normalize(query_x.to(device).float(), dim=1)
    bank_y, query_y = bank_y.to(device).long(), query_y.to(device).long()
    k = min(k, bank.shape[0])
    correct = 0
    for start in range(0, query.shape[0], chunk):
        sims = query[start:start + chunk] @ bank.t()
        top_sim, top_idx = sims.topk(k, dim=1)
        votes = torch.zeros(sims.shape[0], num_classes, device=device)
        votes.scatter_add_(1, bank_y[top_idx], (top_sim / temperature).exp())
        correct += (votes.argmax(dim=1) == query_y[start:start + chunk]).sum().item()
    return correct / max(1, query.shape[0])


def state_checksum(module: nn.Module) -> str:
    """SHA-256 over all parameters and buffers (used to prove the encoder stayed frozen)."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _evaluate_linear(classifier: nn.Linear, x: torch.Tensor, y: torch.Tensor, num_classes: int,
                     batch_size: int = 4096) -> Dict[str, Any]:
    classifier.eval()
    losses, preds = [], []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits = classifier(x[start:start + batch_size])
            losses.append(F.cross_entropy(logits, y[start:start + batch_size], reduction="sum").item())
            preds.append(logits.argmax(1).cpu())
    cm = confusion_matrix(torch.cat(preds), y.cpu(), num_classes)
    return {"loss": sum(losses) / x.shape[0], "top1": top1_accuracy(cm), "macro_f1": macro_f1(cm),
            "per_class_accuracy": per_class_accuracy(cm), "confusion_matrix": cm.tolist()}


def fit_linear_classifier(train_x: torch.Tensor, train_y: torch.Tensor, val_x: Optional[torch.Tensor],
                          val_y: Optional[torch.Tensor], cfg: LinearEvalConfig, num_classes: int,
                          device: torch.device) -> Tuple[nn.Linear, List[Dict[str, float]]]:
    """Train a single fully-connected layer on fixed features. The final epoch is reported
    (no checkpoint selection on the test set)."""
    train_x, train_y = train_x.to(device), train_y.to(device).long()
    if val_x is not None:
        val_x, val_y = val_x.to(device), val_y.to(device).long()
    clf = nn.Linear(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)
    history: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        clf.train()
        order = torch.randperm(train_x.shape[0], generator=gen).to(device)
        total = torch.zeros((), device=device)
        for start in range(0, order.numel(), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            loss = F.cross_entropy(clf(train_x[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach() * idx.numel()
        if epoch % 10 == 0 or epoch == cfg.epochs or epoch == 1:
            row = {"epoch": epoch, "train_loss": (total / order.numel()).item()}
            if val_x is not None:
                row["val_top1"] = _evaluate_linear(clf, val_x, val_y, num_classes)["top1"]
            history.append(row)
    return clf, history


def run_linear_evaluation(model: SimCLRModel, cfg: LinearEvalConfig, train_cfg: TrainConfig,
                          device: torch.device) -> Dict[str, Any]:
    """Frozen encoder -> features -> linear classifier -> test metrics."""
    seed_everything(cfg.seed)
    dataset = cfg.dataset or train_cfg.dataset
    model.to(device).eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    checksum_before = state_checksum(model.encoder)

    train_ds = build_labeled_dataset(dataset, cfg.data_dir, True, cfg.train_subset_size, cfg.subset_seed,
                                     train_cfg.image_size, cfg.download)
    test_ds = build_labeled_dataset(dataset, cfg.data_dir, False, cfg.test_subset_size, cfg.subset_seed + 1,
                                    train_cfg.image_size, cfg.download)
    train_x, train_y = extract_features(model, build_eval_loader(train_ds, 512, cfg.num_workers, device), device)
    test_x, test_y = extract_features(model, build_eval_loader(test_ds, 512, cfg.num_workers, device), device)

    perm = np.random.default_rng(cfg.subset_seed).permutation(train_x.shape[0])
    n_val = int(round(cfg.val_fraction * train_x.shape[0]))
    val_idx, fit_idx = torch.from_numpy(perm[:n_val]), torch.from_numpy(perm[n_val:])
    fit_x, fit_y = train_x[fit_idx], train_y[fit_idx]
    val_x, val_y = (train_x[val_idx], train_y[val_idx]) if n_val > 0 else (None, None)

    mean = std = None
    if cfg.standardize:  # statistics from the fit split only
        mean, std = fit_x.mean(0, keepdim=True), fit_x.std(0, keepdim=True).clamp_min(1e-6)
        fit_x, test_x = (fit_x - mean) / std, (test_x - mean) / std
        if val_x is not None:
            val_x = (val_x - mean) / std

    clf, history = fit_linear_classifier(fit_x, fit_y, val_x, val_y, cfg, NUM_CLASSES, device)
    test = _evaluate_linear(clf, test_x.to(device), test_y.to(device).long(), NUM_CLASSES)

    checksum_after = state_checksum(model.encoder)
    if checksum_before != checksum_after:
        raise RuntimeError("Encoder weights changed during linear evaluation - the encoder must stay frozen.")
    return {
        "test_top1_accuracy": test["top1"], "test_loss": test["loss"], "test_macro_f1": test["macro_f1"],
        "test_per_class_accuracy": test["per_class_accuracy"], "test_confusion_matrix": test["confusion_matrix"],
        "encoder_feature_dim": int(train_x.shape[1]),
        "num_labeled_train_samples_used_to_fit": int(fit_x.shape[0]),
        "num_validation_samples": int(0 if val_x is None else val_x.shape[0]),
        "num_test_samples": int(test_x.shape[0]),
        "training_history": history, "encoder_frozen_verified": True,
        "dataset": dataset, "checkpoint_epoch_config": train_cfg.to_dict(),
    }
