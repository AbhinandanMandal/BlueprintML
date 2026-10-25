"""Validation metrics and visualisation"""
import os

import numpy as np
import torch

from .losses import ReconstructionLoss
from .masking import make_mask


@torch.no_grad()
def evaluate(G, loader, kind, device, rec_loss=None, max_batches=8, seed=123):
    """Masked L2 on the hole + PSNR of the composite image, with fixed (seeded) masks.

    PSNR is computed on the [0, 1] scale. Leaves ``G`` in eval mode (as the notebook does).
    Returns ``(mean_rec_loss, mean_psnr)``.
    """
    rec_loss = rec_loss or ReconstructionLoss().to(device)
    G.eval()
    rng = np.random.default_rng(seed)
    tot_rec = tot_psnr = n = 0
    for i, img in enumerate(loader):
        if i >= max_batches:
            break
        img = img.to(device)
        mask = make_mask(rng, img.size(0), kind, device)
        masked = img * (1 - mask)
        pred = G(masked)
        comp = masked + pred * mask
        B = img.size(0)
        tot_rec += rec_loss(pred, img, mask).item() * B
        mse01 = (((comp - img) * 0.5) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        tot_psnr += (10 * torch.log10(1 / mse01)).sum().item()
        n += B
    return tot_rec / n, tot_psnr / n


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@torch.no_grad()
def save_samples(G, loader, device, path, kind="central", n=6, seed=7, title=""):
    """Save a grid: masked input / raw prediction / composite / ground truth (replaces ``plt.show``)."""
    plt = _plt()
    G.eval()
    rng = np.random.default_rng(seed)
    img = next(iter(loader))[:n].to(device)
    mask = make_mask(rng, img.size(0), kind, device)
    masked = img * (1 - mask)
    pred = G(masked)
    comp = masked + pred * mask

    n = img.size(0)
    to01 = lambda t: (t * 0.5 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    rows = [("Masked input", masked), ("Raw prediction", pred),
            ("Composite", comp), ("Ground truth", img)]
    fig, axes = plt.subplots(len(rows), n, figsize=(2 * n, 2.1 * len(rows)), squeeze=False)
    for r, (name, t) in enumerate(rows):
        arr = to01(t)
        for c in range(n):
            axes[r, c].imshow(arr[c]); axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=9, loc="left")
    fig.suptitle(f"{title} [{kind} mask]")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_history_plot(history, path):
    """Training curves (reconstruction / adversarial / validation PSNR)."""
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(history["rec"], label="train rec"); ax[0].plot(history["val_rec"], label="val rec")
    ax[0].legend(); ax[0].set_title("Reconstruction")
    ax[1].plot(history["loss_D"], label="D"); ax[1].plot(history["adv"], label="G adv")
    ax[1].legend(); ax[1].set_title("Adversarial")
    ax[2].plot(history["val_psnr"]); ax[2].set_title("Val PSNR (dB)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
