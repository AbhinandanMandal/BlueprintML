import math

import pytest
import torch

from simclr.losses import NTXentLoss, nt_xent_loss, positive_indices, similarity_logits
from simclr.models import build_model
from simclr.config import TrainConfig


def reference_nt_xent(z1, z2, tau):
    """Independent, loop-based implementation of paper Eq. (1) in float64."""
    n = z1.shape[0]
    z = [v.double() / v.double().norm() for v in torch.cat([z1, z2])]
    total = 0.0
    for i in range(2 * n):
        j = (i + n) % (2 * n)
        sim = lambda a, b: float(torch.dot(z[a], z[b]))
        numerator = math.exp(sim(i, j) / tau)
        denominator = sum(math.exp(sim(i, k) / tau) for k in range(2 * n) if k != i)
        total += -math.log(numerator / denominator)
    return total / (2 * n)


@pytest.mark.parametrize("n", [2, 3, 8, 33])
@pytest.mark.parametrize("tau", [0.1, 0.5, 1.0])
def test_matches_reference(n, tau):
    torch.manual_seed(n)
    z1, z2 = torch.randn(n, 16), torch.randn(n, 16)
    assert nt_xent_loss(z1, z2, tau).item() == pytest.approx(reference_nt_xent(z1, z2, tau), rel=1e-5)


def test_similarity_matrix_shape_and_masked_diagonal():
    n = 5
    logits = similarity_logits(torch.randn(n, 8), torch.randn(n, 8), 0.5)
    assert logits.shape == (2 * n, 2 * n)
    diag = torch.diagonal(logits)
    assert torch.all(diag == torch.finfo(torch.float32).min)  # self-similarity excluded
    off = logits[~torch.eye(2 * n, dtype=torch.bool)]
    assert torch.all(off > -1 / 0.5 - 1e-4) and torch.all(off < 1 / 0.5 + 1e-4)  # cosine / tau bounds


def test_positive_indices():
    assert positive_indices(3).tolist() == [3, 4, 5, 0, 1, 2]


def test_positive_stays_in_denominator():
    # If the positive were removed from the denominator, identical views would give loss 0.
    z = torch.randn(4, 8)
    assert nt_xent_loss(z, z.clone(), 0.5).item() > 0


def test_aligned_views_have_lower_loss_than_random():
    torch.manual_seed(0)
    z = torch.randn(16, 32)
    aligned = nt_xent_loss(z, z + 0.01 * torch.randn_like(z)).item()
    random = nt_xent_loss(z, torch.randn(16, 32)).item()
    assert aligned < random


def test_loss_is_scale_invariant():
    torch.manual_seed(1)
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    assert nt_xent_loss(z1, z2).item() == pytest.approx(nt_xent_loss(3 * z1, 0.2 * z2).item(), rel=1e-5)


def test_symmetric_in_views():
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    assert nt_xent_loss(z1, z2).item() == pytest.approx(nt_xent_loss(z2, z1).item(), rel=1e-6)


def test_gradients_reach_projection_head_and_encoder():
    torch.manual_seed(0)
    model = build_model(TrainConfig(projection_hidden_dim=128)).train()
    x = torch.randn(8, 3, 32, 32)  # 4 images x 2 views
    _, z = model(x)
    loss = nt_xent_loss(z[:4], z[4:], 0.5)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in list(model.projector.named_parameters()) + list(model.encoder.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        nt_xent_loss(torch.randn(4), torch.randn(4))
    with pytest.raises(ValueError):
        nt_xent_loss(torch.randn(4, 8), torch.randn(5, 8))
    with pytest.raises(ValueError):
        nt_xent_loss(torch.randn(1, 8), torch.randn(1, 8))
    with pytest.raises(ValueError):
        nt_xent_loss(torch.randn(4, 8), torch.randn(4, 8), temperature=0)
    with pytest.raises(ValueError):
        NTXentLoss(-1.0)


def test_module_wrapper_and_half_precision_input():
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    assert NTXentLoss(0.5)(z1, z2).item() == pytest.approx(nt_xent_loss(z1, z2, 0.5).item())
    assert torch.isfinite(nt_xent_loss(z1.bfloat16(), z2.bfloat16()))  # computed in fp32 internally
