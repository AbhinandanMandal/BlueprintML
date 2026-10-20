import pytest
import torch
import torch.nn.functional as F

from moco_reproduction.config import MoCoConfig
from moco_reproduction.losses import moco_logits, info_nce_loss
from moco_reproduction.models.encoder import MoCoEncoder
from moco_reproduction.models.moco import MoCo
from moco_reproduction.train import build_optimizer, train_step, save_checkpoint, load_checkpoint, build_scheduler

N, K, C = 8, 32, 16


@pytest.fixture()
def model():
    torch.manual_seed(0)
    enc = MoCoEncoder("resnet18", C, cifar_stem=True)
    return MoCo(enc, C, K, 0.99, 0.07)


def batch():
    return torch.randn(N, 3, 32, 32), torch.randn(N, 3, 32, 32)


def test_shapes_normalisation_and_targets(model):
    xq, xk = batch()
    out = model(xq, xk)
    assert out.q.shape == (N, C) and out.k.shape == (N, C)
    assert out.logits.shape == (N, 1 + K)
    assert out.targets.shape == (N,) and torch.all(out.targets == 0)
    assert torch.allclose(out.q.norm(dim=1), torch.ones(N), atol=1e-4)
    assert torch.allclose(out.k.norm(dim=1), torch.ones(N), atol=1e-4)
    assert torch.allclose(model.queue.norm(dim=1), torch.ones(K), atol=1e-4)
    assert torch.isfinite(out.logits).all() and torch.isfinite(model.loss(out))
    # positive logit/negative logits have the documented shapes
    assert (out.logits[:, :1]).shape == (N, 1) and (out.logits[:, 1:]).shape == (N, K)


def test_gradient_flow_and_optimizer_step(model):
    cfg = MoCoConfig(batch_size=N, subset_size=N, queue_size=K, lr=0.1)
    opt = build_optimizer(cfg, model)
    xq, xk = batch()
    before_q = [p.detach().clone() for p in model.encoder_q.parameters()]
    out = model(xq, xk)
    model.loss(out).backward()
    assert all(p.grad is not None for p in model.encoder_q.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for p in model.encoder_q.parameters())
    assert all(p.grad is None for p in model.encoder_k.parameters())
    assert model.queue.grad is None and not model.queue.requires_grad
    assert not out.k.requires_grad
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before_q, model.encoder_q.parameters()))
    assert all(a.requires_grad is False for a in model.encoder_k.parameters())


def test_logits_construction_on_controlled_example():
    q = F.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    k = F.normalize(torch.tensor([[1.0, 0.0]]), dim=1)                # identical -> l_pos = 1
    queue = F.normalize(torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]), dim=1)
    tau = 0.5
    logits, targets = moco_logits(q, k, queue, tau)
    assert logits.shape == (1, 4) and targets.item() == 0
    expected = torch.tensor([[1.0, 0.0, -1.0, 2 ** -0.5]]) / tau
    assert torch.allclose(logits, expected, atol=1e-6)
    manual = -torch.log(torch.exp(logits[0, 0]) / torch.exp(logits[0]).sum())
    assert torch.allclose(info_nce_loss(logits, targets), manual, atol=1e-6)


def test_loss_decreases_when_positive_is_closer_and_query_gets_gradient():
    queue = F.normalize(torch.randn(16, 4), dim=1)
    q = F.normalize(torch.randn(2, 4), dim=1).requires_grad_(True)
    near, far = q.detach().clone(), -q.detach().clone()
    l_near = info_nce_loss(*moco_logits(q, near, queue, 0.07))
    l_far = info_nce_loss(*moco_logits(q, far, queue, 0.07))
    assert l_near < l_far
    l_near.backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_train_step_updates_queue_and_keys(model):
    cfg = MoCoConfig(batch_size=N, subset_size=N, queue_size=K, lr=0.05)
    opt = build_optimizer(cfg, model)
    xq, xk = batch()
    k_before = [p.detach().clone() for p in model.encoder_k.parameters()]
    q_before = [p.detach().clone() for p in model.encoder_q.parameters()]
    m = train_step(model, opt, xq, xk)
    assert m["queue_ptr"] == N and int(model.queue_ptr) == N
    assert torch.isfinite(torch.tensor(m["loss"]))
    # At step 1 the key params are updated from the (identical) query params -> unchanged
    assert all(torch.allclose(a, b) for a, b in zip(k_before, model.encoder_k.parameters()))
    m2 = train_step(model, opt, xq, xk)
    # now f_q has moved, so f_k must move toward it but by a small amount
    moved = [(a - b).abs().max().item() for a, b in zip(q_before, model.encoder_q.parameters())]
    assert max(moved) > 0
    assert any(not torch.equal(a, b) for a, b in zip(k_before, model.encoder_k.parameters()))


def test_checkpoint_roundtrip(tmp_path, model):
    cfg = MoCoConfig(batch_size=N, subset_size=N, queue_size=K, lr=0.05, epochs=2, lr_schedule="step")
    opt, sched = build_optimizer(cfg, model), build_scheduler(cfg, build_optimizer(cfg, model))
    xq, xk = batch()
    train_step(model, opt, xq, xk)
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, model, opt, None, 1, 1, cfg, [{"step": 1}])
    fresh = MoCo(MoCoEncoder("resnet18", C, True), C, K, 0.99, 0.07)
    fresh_opt = build_optimizer(cfg, fresh)
    state = load_checkpoint(p, fresh, fresh_opt)
    assert state["epoch"] == 1 and state["global_step"] == 1
    assert torch.equal(fresh.queue, model.queue) and int(fresh.queue_ptr) == int(model.queue_ptr)
    for a, b in zip(model.encoder_q.parameters(), fresh.encoder_q.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(model.encoder_k.parameters(), fresh.encoder_k.parameters()):
        assert torch.equal(a, b)
    wrong = MoCo(MoCoEncoder("resnet18", C, True), C, K * 2, 0.99, 0.07)
    with pytest.raises(ValueError):
        load_checkpoint(p, wrong)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "nope.pt"), fresh)


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        MoCoConfig(batch_size=64, subset_size=32).validate()
    with pytest.raises(ValueError):
        MoCoConfig(momentum=1.0).validate()
    with pytest.raises(ValueError):
        MoCoConfig(temperature=0).validate()


def test_shuffle_bn_emulation_restores_key_order():
    """In eval mode BN is per-sample, so shuffled+unshuffled keys must equal plain keys."""
    torch.manual_seed(0)
    enc = MoCoEncoder("resnet18", C, True)
    m = MoCo(enc, C, K, 0.99, 0.07, bn_groups=2)
    m.eval()
    x = torch.randn(N, 3, 32, 32)
    assert torch.allclose(m._encode_keys(x), m.encoder_k(x), atol=1e-5)
