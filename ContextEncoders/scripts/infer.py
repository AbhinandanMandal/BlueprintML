"""Inpaint user-provided images with a trained checkpoint.

Example:
    python scripts/infer.py --checkpoint outputs/run1/context_encoder_last.pt \
        --images face1.jpg face2.png --mask central --output-dir results
Writes one comparison PNG: rows = masked input / raw prediction / composite / ground truth.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from src.context_encoders.dataset import build_transform  # noqa: E402
from src.context_encoders.masking import MASK_FNS, make_mask  # noqa: E402
from src.context_encoders.model import ContextEncoder  # noqa: E402
from src.context_encoders.utils import get_device, load_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--mask", default="central", choices=sorted(MASK_FNS))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = get_device(a.device)
    G = ContextEncoder().to(device)
    load_checkpoint(a.checkpoint, G, map_location=device)
    G.eval()

    tf = build_transform(128)
    imgs = []
    for path in a.images:
        if not os.path.isfile(path):
            sys.exit(f"Image not found: '{path}'")
        imgs.append(tf(Image.open(path).convert("RGB")))
    img = torch.stack(imgs).to(device)

    rng = np.random.default_rng(a.seed)
    mask = make_mask(rng, img.size(0), a.mask, device)
    masked = img * (1 - mask)
    with torch.no_grad():
        pred = G(masked)
    comp = masked + pred * mask

    rows = torch.cat([masked, pred, comp, img]) * 0.5 + 0.5          # un-normalise to [0, 1]
    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, f"inpainting_{a.mask}.png")
    save_image(rows.clamp(0, 1), out, nrow=img.size(0))
    print(f"Saved {out}  (rows: masked input / raw prediction / composite / ground truth)")


if __name__ == "__main__":
    main()
