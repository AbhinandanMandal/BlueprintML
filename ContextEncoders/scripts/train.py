"""Train a Context Encoder on CelebA.

Example:
    python scripts/train.py --data-dir /path/to/img_align_celeba --output-dir outputs/run1
Defaults reproduce the notebook's hyper-parameters (see configs/default.py).
"""
import argparse
import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

import torch 

from configs.default import Config 
from src.context_encoders.dataset import make_loaders  
from src.context_encoders.engine import fit  
from src.context_encoders.masking import MASK_FNS  
from src.context_encoders.model import ContextEncoder, Discriminator 
from src.context_encoders.utils import get_device, set_seed  


def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with CelebA images (img_align_celeba)")
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--adv-start-epoch", type=int, default=d.adv_start_epoch)
    p.add_argument("--lambda-rec", type=float, default=d.lambda_rec)
    p.add_argument("--lambda-adv", type=float, default=d.lambda_adv)
    p.add_argument("--train-size", type=int, default=d.train_size)
    p.add_argument("--val-size", type=int, default=d.val_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--mask-kinds", nargs="+", default=d.mask_kinds, choices=sorted(MASK_FNS))
    p.add_argument("--max-steps", type=int, default=d.max_steps, help="steps per epoch (smoke tests)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--split-seed", type=int, default=d.split_seed)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--overwrite", action="store_true", help="allow overwriting an existing checkpoint in output-dir")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = Config(**{k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__})
    ckpt = os.path.join(cfg.output_dir, cfg.ckpt_name)
    if os.path.exists(ckpt) and not a.overwrite:
        sys.exit(f"'{ckpt}' already exists. Use a new --output-dir or pass --overwrite.")

    device = get_device(a.device)
    torch.backends.cudnn.benchmark = True
    set_seed(cfg.seed)

    train_loader, val_loader, (tr_idx, va_idx) = make_loaders(
        cfg.data_dir, cfg.train_size, cfg.val_size, cfg.batch_size, cfg.num_workers,
        cfg.image_size, cfg.split_seed, pin_memory=device.type == "cuda")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    with open(os.path.join(cfg.output_dir, "split_indices.json"), "w") as f:
        json.dump({"train": tr_idx.tolist(), "val": va_idx.tolist()}, f)

    G, D = ContextEncoder().to(device), Discriminator().to(device)
    print(f"device={device} | G params {sum(p.numel() for p in G.parameters())/1e6:.3f}M | "
          f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")
    fit(G, D, train_loader, val_loader, cfg, device)
    print(f"Done. Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
