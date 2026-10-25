"""Reconstruction, adversarial and joint losses."""
import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """Masked L2 loss.

    pred, target: (B, C, H, W);  mask: (B, 1, H, W), 1 = hole (region to predict), 0 = known.
    Per-sample squared error over the hole, divided by (#hole pixels * C + eps), then batch-mean.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask):
        diff = mask * (target - pred)                      # M * (x - F(...))
        sq_err = diff.pow(2).sum(dim=(1, 2, 3))            # squared L2 norm per sample
        norm = mask.sum(dim=(1, 2, 3)) * pred.size(1) + self.eps
        return (sq_err / norm).mean()


class AdversarialLoss(nn.Module):
    """D maximizes log D(x) + log(1 - D(F(...))); equivalently minimizes BCE."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def d_loss(self, d_real_logits, d_fake_logits):
        real = self.bce(d_real_logits, torch.ones_like(d_real_logits))
        fake = self.bce(d_fake_logits, torch.zeros_like(d_fake_logits))
        return real + fake

    def g_loss(self, d_fake_logits):
        # Minimize -log D(F(...)) instead of log(1 - D(F(...)))
        return self.bce(d_fake_logits, torch.ones_like(d_fake_logits))


class JointLoss(nn.Module):
    """L = lambda_rec * L_rec + lambda_adv * L_adv (generator side)."""

    def __init__(self, lambda_rec=0.999, lambda_adv=0.001):
        super().__init__()
        self.lambda_rec, self.lambda_adv = lambda_rec, lambda_adv
        self.rec = ReconstructionLoss()
        self.adv = AdversarialLoss()

    def forward(self, pred, target, mask, d_fake_logits):
        l_rec = self.rec(pred, target, mask)
        l_adv = self.adv.g_loss(d_fake_logits)
        total = self.lambda_rec * l_rec + self.lambda_adv * l_adv
        return total, l_rec, l_adv
