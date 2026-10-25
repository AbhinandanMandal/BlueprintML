"""CPU smoke test: tiny synthetic dataset, a few steps, checkpoint round trip.
--------------------------------------------------------------------------------
Following ``tests`` are refracted from ``CLAUDE`` for industry grade practices
"""
import dataclasses
import os

import torch

from configs.default import Config
from src.context_encoders.engine import fit
from src.context_encoders.model import ContextEncoder, Discriminator
from src.context_encoders.utils import load_checkpoint, set_seed


def test_short_training_run_and_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    data = [torch.rand(4, 3, 128, 128) * 2 - 1 for _ in range(2)]    # 2 batches of 4
    cfg = Config(epochs=2, adv_start_epoch=1, max_steps=2, output_dir=str(tmp_path),
                 mask_kinds=["central", "block"], eval_max_batches=1, vis_every=1)
    G, D = ContextEncoder(), Discriminator()
    g0 = {k: v.clone() for k, v in G.state_dict().items()}
    hist = fit(G, D, data, data, cfg, torch.device("cpu"), log=lambda *_: None)

    assert all(torch.isfinite(torch.tensor(v)).all() for h in hist.values() for v in [h])
    assert hist["loss_D"][0] == 0.0 and hist["loss_D"][1] > 0.0      # epoch 1 warm-up, epoch 2 adversarial
    assert any(not torch.equal(g0[k], v) for k, v in G.state_dict().items())   # optimizer updated G

    ckpt = os.path.join(tmp_path, cfg.ckpt_name)
    assert set(torch.load(ckpt, weights_only=True)) == {"G", "D", "epoch"}
    G2, D2 = ContextEncoder(), Discriminator()
    assert load_checkpoint(ckpt, G2, D2) == 2
    for (k, a), (_, b) in zip(G.state_dict().items(), G2.state_dict().items()):
        assert torch.equal(a, b), k
