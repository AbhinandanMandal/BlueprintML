"""Compare against the ORIGINAL notebook.
"""
import copy
import json
import os
import re
import types

import numpy as np
import pytest
import torch

NB = os.environ.get("CONTEXT_ENCODERS_NOTEBOOK")
pytestmark = pytest.mark.skipif(not NB or not os.path.isfile(NB or ""), reason="set CONTEXT_ENCODERS_NOTEBOOK")


def _cell(nb, marker):
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and marker in src:
            return src
    raise KeyError(marker)


@pytest.fixture(scope="module")
def orig():
    nb = json.load(open(NB))
    ns = {}
    for marker in ("def mask_central", "class Encoder(nn.Module)", "class ReconstructionLoss", "class Discriminator"):
        exec(_cell(nb, marker), ns)
    ns["_nb"] = nb
    return types.SimpleNamespace(**ns)


def test_model_weights_and_outputs_identical(orig):
    from src.context_encoders.model import ContextEncoder, Discriminator
    torch.manual_seed(0); g_o, d_o = orig.ContextEncoder(), orig.Discriminator()
    torch.manual_seed(0); g_n, d_n = ContextEncoder(), Discriminator()
    assert list(g_o.state_dict()) == list(g_n.state_dict())
    assert list(d_o.state_dict()) == list(d_n.state_dict())
    g_n.load_state_dict(g_o.state_dict()); d_n.load_state_dict(d_o.state_dict())   # strict: names + shapes
    x = torch.randn(4, 3, 128, 128)
    assert torch.equal(g_o(x), g_n(x)) and torch.equal(d_o(x), d_n(x))
    # same-seed init is bit-identical too
    for (k, a), (_, b) in zip(g_o.state_dict().items(), g_n.state_dict().items()):
        assert torch.equal(a, b), k


@pytest.mark.parametrize("kind", ["central", "block", "region"])
def test_masks_identical(orig, kind):
    from src.context_encoders.masking import MASK_FNS
    a = getattr(orig, f"mask_{kind}")(np.random.default_rng(5), 3)
    b = MASK_FNS[kind](np.random.default_rng(5), 3)
    np.testing.assert_array_equal(a, b)


def test_losses_identical(orig):
    from src.context_encoders.losses import JointLoss
    torch.manual_seed(1)
    pred, tgt = torch.randn(4, 3, 128, 128), torch.randn(4, 3, 128, 128)
    mask = torch.from_numpy(orig.mask_block(np.random.default_rng(2), 4)).unsqueeze(1)
    logits = torch.randn(4, 1)
    o = orig.JointLoss(0.999, 0.001)(pred, tgt, mask, logits)
    n = JointLoss(0.999, 0.001)(pred, tgt, mask, logits)
    for a, b in zip(o, n):
        assert torch.equal(a, b)


def test_training_trajectory_identical(orig, tmp_path):
    """Run the notebook's own training cell and ``fit`` side by side on identical synthetic data."""
    import matplotlib; matplotlib.use("Agg")
    from configs.default import Config
    from src.context_encoders.engine import fit
    from src.context_encoders.model import ContextEncoder, Discriminator

    torch.manual_seed(123)
    # batch of 6: the notebook's show_samples hard-codes n=6 columns
    data = [torch.rand(6, 3, 128, 128) * 2 - 1 for _ in range(2)]
    EPOCHS, ADV, STEPS = 3, 1, 2

    src = _cell(orig._nb, "ADV_START_EPOCH = 2")
    src = re.sub(r"^EPOCHS\s*=.*$", f"EPOCHS = {EPOCHS}", src, flags=re.M)
    src = re.sub(r"^ADV_START_EPOCH\s*=.*$", f"ADV_START_EPOCH = {ADV}", src, flags=re.M)
    src = re.sub(r"^MAX_STEPS\s*=.*$", f"MAX_STEPS = {STEPS}", src, flags=re.M)
    src = re.sub(r"^CKPT_DIR\s*=.*$", f"CKPT_DIR = r'{tmp_path / 'nb'}'", src, flags=re.M)
    src = src.replace("plt.show()", "plt.close('all')")
    ns = dict(vars(orig)); ns.update(train_loader=data, test_loader=data)
    exec(src, ns)

    cfg = Config(epochs=EPOCHS, adv_start_epoch=ADV, max_steps=STEPS, output_dir=str(tmp_path / "new"))
    torch.manual_seed(cfg.seed)
    G, D = ContextEncoder(), Discriminator()
    hist = fit(G, D, data, data, cfg, torch.device("cpu"), log=lambda *_: None)

    for k in ns["history"]:
        np.testing.assert_allclose(hist[k], ns["history"][k], rtol=1e-6, atol=1e-7, err_msg=k)
    for (k, a), (_, b) in zip(ns["G"].state_dict().items(), G.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k
