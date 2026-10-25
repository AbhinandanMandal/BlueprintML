"""Mask generators (1 = hole to predict, 0 = known pixel). Logic copied verbatim from the notebook.

``rng`` is a ``numpy.random.Generator``. ``mask_central`` ignores it (kept for a uniform signature).
All functions return a float32 array of shape (B, H, W).
"""
import numpy as np
import torch

MASK_KINDS = ("central", "block", "region")


def mask_central(rng, B, H=128, W=128):
    """Fixed central square (H/2 x W/2), i.e. 25% of the image."""
    m = np.zeros((B, H, W), dtype=np.float32)
    h, w = H // 2, W // 2
    y, x = (H - h) // 2, (W - w) // 2
    m[:, y:y + h, x:x + w] = 1.0
    return m


def mask_block(rng, B, H=128, W=128, max_cover=None):
    """3-6 random rectangles per image, total coverage capped at ``max_cover`` (default H*W/4)."""
    if max_cover is None:
        max_cover = (H * W) // 4
    m = np.zeros((B, H, W), dtype=np.float32)
    for b in range(B):
        for _ in range(int(rng.integers(3, 7))):
            h = int(rng.integers(max(2, H // 32), max(3, H // 8)))
            w = int(rng.integers(max(2, W // 32), max(3, W // 8)))

            y = int(rng.integers(0, H - h + 1))
            x = int(rng.integers(0, W - w + 1))

            candidate = m[b].copy()
            candidate[y:y + h, x:x + w] = 1.0

            if candidate.sum() <= max_cover:
                m[b] = candidate
    return m


def mask_region(rng, B, H=128, W=128, max_cover=None):
    """Random-walk brush producing an irregular region, coverage capped at ``max_cover``."""
    if max_cover is None:
        max_cover = (H * W) // 4
    m = np.zeros((B, H, W), dtype=np.float32)
    brush = max(3, min(H, W) // 32)

    for b in range(B):
        target = int(rng.integers(max(1, max_cover // 2), max_cover + 1))
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        covered = 0
        for _ in range(H * W * 2):
            y0, y1 = max(0, y - brush // 2), min(H, y + brush // 2 + 1)
            x0, x1 = max(0, x - brush // 2), min(W, x + brush // 2 + 1)
            patch = m[b, y0:y1, x0:x1]                 # view into m
            added = int((patch == 0).sum())            # newly covered pixels
            if covered + added <= max_cover:
                patch[:] = 1.0
                covered += added
                if covered >= target:
                    break
            y = int(np.clip(y + rng.integers(-brush, brush + 1), 0, H - 1))
            x = int(np.clip(x + rng.integers(-brush, brush + 1), 0, W - 1))
    return m


MASK_FNS = {"central": mask_central,
            "block": mask_block, "region": mask_region}


def make_mask(rng, B, kind, device="cpu"):
    """Return a (B,1,H,W) float tensor on ``device``, 1 = hole. Uses the default 128x128 size, as the notebook."""
    if kind not in MASK_FNS:
        raise ValueError(
            f"Unknown mask kind '{kind}'. Choose from {sorted(MASK_FNS)}.")
    return torch.from_numpy(MASK_FNS[kind](rng, B)).unsqueeze(1).to(device)
