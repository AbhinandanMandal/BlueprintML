"""Training loop, split into an epoch function and an outer ``fit``."""
import json
import os
import time

import numpy as np
import torch

from .evaluate import evaluate, save_history_plot, save_samples
from .losses import JointLoss
from .masking import make_mask
from .utils import save_checkpoint


def train_one_epoch(G, D, loader, joint_loss, opt_G, opt_D, rng, device, epoch,
                    adv_start_epoch=2, mask_kinds=("central", "block", "region"), max_steps=None):
    """One epoch. Returns mean ``loss_D, loss_G, rec, adv`` """
    G.train(); D.train()
    use_adv = epoch > adv_start_epoch
    adv_loss, rec_loss = joint_loss.adv, joint_loss.rec
    sums = dict(loss_D=0.0, loss_G=0.0, rec=0.0, adv=0.0)
    steps = 0

    for step, img in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        img = img.to(device, non_blocking=True)
        B = img.size(0)

        kind = mask_kinds[rng.integers(len(mask_kinds))]       # random mask type per batch
        mask = make_mask(rng, B, kind, device)
        masked = img * (1 - mask)
        pred = G(masked)                                       # full-size output

        if use_adv:
            # Discriminator step
            loss_D = adv_loss.d_loss(D(img), D(pred.detach()))
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # Generator step: lambda_rec * L_rec + lambda_adv * L_adv
            loss_G, l_rec, l_adv = joint_loss(pred, img, mask, D(pred))
        else:
            # warm-up: reconstruction only
            l_rec = rec_loss(pred, img, mask)
            loss_G = l_rec
            loss_D = l_adv = torch.zeros((), device=device)

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        sums["loss_D"] += loss_D.item(); sums["loss_G"] += loss_G.item()
        sums["rec"] += l_rec.item();     sums["adv"] += l_adv.item()
        steps += 1

    if steps == 0:
        raise RuntimeError("The training loader yielded no batches.")
    return {k: v / steps for k, v in sums.items()}


def fit(G, D, train_loader, val_loader, cfg, device, log=print):
    """Full training run. Returns the ``history`` dict"""
    joint_loss = JointLoss(cfg.lambda_rec, cfg.lambda_adv).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))

    history = {"loss_D": [], "loss_G": [], "rec": [], "adv": [], "val_rec": [], "val_psnr": []}
    rng = np.random.default_rng(cfg.mask_seed)
    ckpt_path = os.path.join(cfg.output_dir, cfg.ckpt_name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        use_adv = epoch > cfg.adv_start_epoch
        m = train_one_epoch(G, D, train_loader, joint_loss, opt_G, opt_D, rng, device, epoch,
                            cfg.adv_start_epoch, cfg.mask_kinds, cfg.max_steps)
        for k, v in m.items():
            history[k].append(v)

        val = {k: evaluate(G, val_loader, k, device, joint_loss.rec, cfg.eval_max_batches, cfg.eval_seed)
               for k in cfg.mask_kinds}
        history["val_rec"].append(float(np.mean([v[0] for v in val.values()])))
        history["val_psnr"].append(float(np.mean([v[1] for v in val.values()])))

        log(f"Epoch {epoch:02d}/{cfg.epochs} | {time.time()-t0:.0f}s | adv={'on ' if use_adv else 'off'} | "
            f"D {history['loss_D'][-1]:.3f} | G {history['loss_G'][-1]:.4f} | "
            f"rec {history['rec'][-1]:.4f} | adv {history['adv'][-1]:.3f} | "
            f"val rec {history['val_rec'][-1]:.4f} | val PSNR {history['val_psnr'][-1]:.2f} dB")

        if epoch % cfg.vis_every == 0 or epoch == cfg.epochs:
            save_samples(G, val_loader, device, os.path.join(cfg.output_dir, "samples", f"epoch_{epoch:02d}_central.png"),
                         kind="central", seed=cfg.vis_seed, title=f"Epoch {epoch}")

        save_checkpoint(ckpt_path, G, D, epoch)
        with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    save_history_plot(history, os.path.join(cfg.output_dir, "curves.png"))
    for k in cfg.mask_kinds:
        save_samples(G, val_loader, device, os.path.join(cfg.output_dir, "samples", f"final_{k}.png"),
                     kind=k, seed=cfg.vis_seed, title="Final")
    return history
