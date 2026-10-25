import torch

from src.context_encoders.losses import AdversarialLoss, JointLoss, ReconstructionLoss


def test_reconstruction_loss_value_and_mask_only():
    pred, target = torch.zeros(2, 3, 4, 4), torch.ones(2, 3, 4, 4)
    mask = torch.zeros(2, 1, 4, 4); mask[:, :, :2, :2] = 1          # 4 hole pixels
    loss = ReconstructionLoss()(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)         # (4*3*1)/(4*3)
    pred2 = pred.clone(); pred2[:, :, 2:, 2:] = 99                   # outside the hole -> no effect
    assert torch.isclose(ReconstructionLoss()(pred2, target, mask), loss)


def test_losses_finite_and_differentiable():
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    target, mask = torch.randn(2, 3, 8, 8), (torch.rand(2, 1, 8, 8) > 0.5).float()
    logits = torch.randn(2, 1, requires_grad=True)
    total, rec, adv = JointLoss()(pred, target, mask, logits)
    total.backward()
    assert torch.isfinite(total) and pred.grad is not None and logits.grad is not None
    a = AdversarialLoss()
    assert torch.isfinite(a.d_loss(torch.randn(4, 1), torch.randn(4, 1)))
