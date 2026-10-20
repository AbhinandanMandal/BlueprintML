import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from moco_reproduction.models.moco import MoCo


class Tiny(nn.Module):
    def __init__(self, d_in=6, d_out=4):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)


def make(m=0.9, K=8, C=4):
    return MoCo(Tiny(6, C), C, K, m, 0.07)


def test_key_encoder_starts_identical_and_frozen():
    model = make()
    for pq, pk in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        assert torch.equal(pq, pk)
        assert not pk.requires_grad and pq.requires_grad


def test_momentum_update_matches_formula():
    m = 0.9
    model = make(m)
    with torch.no_grad():
        for p in model.encoder_q.parameters():
            p.fill_(2.0)
        for p in model.encoder_k.parameters():
            p.fill_(10.0)
    old_q = [p.detach().clone() for p in model.encoder_q.parameters()]
    old_k = [p.detach().clone() for p in model.encoder_k.parameters()]
    model.momentum_update()
    for p_q, p_k, oq, ok in zip(model.encoder_q.parameters(), model.encoder_k.parameters(), old_q, old_k):
        assert torch.allclose(p_k, m * ok + (1 - m) * oq)   # 0.9*10 + 0.1*2 = 9.2
        assert torch.allclose(p_k, torch.full_like(p_k, 9.2))
        assert torch.equal(p_q, oq)                          # query encoder untouched
        assert not p_k.requires_grad and p_k.grad is None


def test_m_zero_copies_query_and_high_m_barely_moves():
    model = make(0.0)
    with torch.no_grad():
        for p in model.encoder_q.parameters():
            p.add_(1.0)
    model.momentum_update()
    for pq, pk in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        assert torch.allclose(pq, pk)
