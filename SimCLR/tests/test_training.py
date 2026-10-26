import math

import pytest
import torch
import torch.nn as nn

from simclr.config import LinearEvalConfig, TrainConfig
from simclr.datasets import build_pretrain_loader
from simclr.engine import train_one_epoch
from simclr.evaluation import fit_linear_classifier, knn_accuracy, run_linear_evaluation, state_checksum
from simclr.losses import nt_xent_loss
from simclr.metrics import confusion_matrix, macro_f1, per_class_accuracy, top1_accuracy
from simclr.models import build_model
from simclr.optimizers import LARS, WarmupCosineScheduler, build_optimizer


# ---------------------------------------------------------------- scheduler / optimizers
def test_scheduler_warmup_peak_and_cosine_end():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = WarmupCosineScheduler(opt, base_lr=1.0, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)  # linear warm-up to the peak
    assert lrs[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # monotone cosine decay
    assert lrs[-1] < 1e-3                                    # last used step is ~0
    assert sched.lr_at(100) == pytest.approx(0.0, abs=1e-12)  # horizon reaches min_lr exactly


def test_scheduler_state_roundtrip():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s1 = WarmupCosineScheduler(opt, 1.0, 50, 5)
    for _ in range(17):
        s1.step()
    s2 = WarmupCosineScheduler(opt, 1.0, 50, 5)
    s2.load_state_dict(s1.state_dict())
    assert s2.last_lr == pytest.approx(s1.last_lr)


def test_scheduler_rejects_bad_horizon():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    with pytest.raises(ValueError):
        WarmupCosineScheduler(opt, 1.0, total_steps=5, warmup_steps=5)


def test_lars_minimises_quadratic_and_exclusion_skips_trust_ratio():
    w = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    opt = LARS([w], lr=1.0, momentum=0.9, trust_coefficient=0.1)
    start = (w ** 2).sum().item()
    for _ in range(200):
        opt.zero_grad()
        (w ** 2).sum().backward()
        opt.step()
    assert (w ** 2).sum().item() < start * 0.1

    b = torch.nn.Parameter(torch.tensor([1.0]))
    opt = LARS([{"params": [b], "lars_exclude": True, "weight_decay": 0.5}], lr=0.1, momentum=0.0)
    b.grad = torch.tensor([2.0])
    opt.step()
    assert b.item() == pytest.approx(1.0 - 0.1 * 2.0)  # plain SGD step: no wd, no trust scaling


@pytest.mark.parametrize("name", ["sgd", "lars"])
def test_build_optimizer_groups(name):
    cfg = TrainConfig(optimizer=name, projection_hidden_dim=64)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    assert opt.param_groups[0]["weight_decay"] == cfg.wd and opt.param_groups[1]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
    assert opt.param_groups[0]["lr"] == pytest.approx(cfg.lr)


def test_learning_rate_scales_linearly_with_batch_size():
    assert TrainConfig(batch_size=128, base_lr=0.3).lr == pytest.approx(0.15)


# ---------------------------------------------------------------- full optimisation steps
def _one_step_setup(make_cfg, **kw):
    cfg = make_cfg(**kw)
    model = build_model(cfg)
    loader = build_pretrain_loader(cfg, torch.device("cpu"))
    opt = build_optimizer(model, cfg)
    sched = WarmupCosineScheduler(opt, cfg.lr, cfg.epochs * len(loader), 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return cfg, model, loader, opt, sched, scaler


def test_one_full_optimisation_step_updates_weights_and_is_finite(make_cfg):
    cfg, model, loader, opt, sched, scaler = _one_step_setup(make_cfg)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items() if v.dtype.is_floating_point}
    v1, v2 = next(iter(loader))
    model.train()
    _, z = model(torch.cat([v1, v2]))
    loss = nt_xent_loss(z[:16], z[16:], cfg.temperature)
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    changed = sum(not torch.equal(before[k], v) for k, v in model.encoder.state_dict().items() if k in before)
    assert changed > 0


def test_train_one_epoch_runs_and_reports_stats(make_cfg):
    cfg, model, loader, opt, sched, scaler = _one_step_setup(make_cfg)
    stats = train_one_epoch(model, loader, opt, sched, scaler, cfg, torch.device("cpu"), use_amp=False)
    assert math.isfinite(stats["loss"]) and stats["steps"] == len(loader) == 4
    assert stats["images"] == 64 and stats["images_per_sec"] > 0
    assert sched.step_count == 4  # scheduler advanced once per iteration


def test_grad_clip_path_runs(make_cfg):
    cfg, model, loader, opt, sched, scaler = _one_step_setup(make_cfg, grad_clip=1.0)
    assert math.isfinite(train_one_epoch(model, loader, opt, sched, scaler, cfg, torch.device("cpu"), False)["loss"])


def test_model_can_optimise_objective_on_fixed_batch(make_cfg):
    """Overfitting/debugging test: loss on a fixed pair of views must fall."""
    cfg = make_cfg(base_lr=1.0, batch_size=16, optimizer="sgd")
    torch.manual_seed(0)
    model = build_model(cfg).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.randn(32, 3, 32, 32)  # 16 images x 2 views, fixed
    losses = []
    for _ in range(25):
        _, z = model(x)
        loss = nt_xent_loss(z[:16], z[16:], 0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(l) for l in losses)
    assert min(losses[-5:]) < losses[0] - 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP/GradScaler test")
def test_amp_step_on_cuda(make_cfg):  # pragma: no cover - not runnable on CPU-only machines
    cfg, model, loader, opt, sched, scaler = _one_step_setup(make_cfg, device="cuda", amp=True)
    model.cuda()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    stats = train_one_epoch(model, loader, opt, sched, scaler, cfg, torch.device("cuda"), use_amp=True)
    assert math.isfinite(stats["loss"])


# ---------------------------------------------------------------- metrics
def test_metrics_against_hand_computed_example():
    preds = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0])
    target = torch.tensor([0, 0, 1, 2, 2, 2, 1, 1])
    cm = confusion_matrix(preds, target, 3)
    assert cm.tolist() == [[2, 0, 0], [1, 1, 1], [0, 1, 2]]
    assert top1_accuracy(cm) == pytest.approx(5 / 8)
    assert per_class_accuracy(cm) == pytest.approx([1.0, 1 / 3, 2 / 3])
    assert macro_f1(cm) == pytest.approx((0.8 + 0.4 + 2 / 3) / 3)


def test_macro_f1_matches_sklearn_if_installed():
    sk = pytest.importorskip("sklearn.metrics")
    g = torch.Generator().manual_seed(0)
    t, p = torch.randint(0, 5, (200,), generator=g), torch.randint(0, 5, (200,), generator=g)
    assert macro_f1(confusion_matrix(p, t, 5)) == pytest.approx(sk.f1_score(t, p, average="macro"))


def test_confusion_matrix_validates_range():
    with pytest.raises(ValueError):
        confusion_matrix(torch.tensor([3]), torch.tensor([0]), 3)


# ---------------------------------------------------------------- evaluation
def test_linear_classifier_learns_and_parameters_update():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (400,), generator=g)
    x = torch.randn(400, 8, generator=g) + 3 * (2 * y.float() - 1).unsqueeze(1)
    cfg = LinearEvalConfig(checkpoint="x", epochs=30, batch_size=64, lr=1e-2)
    torch.manual_seed(0)
    init = nn.Linear(8, 2).weight.clone()
    torch.manual_seed(0)
    clf, history = fit_linear_classifier(x, y, x, y, cfg, 2, torch.device("cpu"))
    assert not torch.equal(init, clf.weight.detach())
    assert history[-1]["val_top1"] > 0.95


def test_knn_accuracy_on_separable_features():
    y = torch.arange(3).repeat_interleave(20)
    x = torch.nn.functional.one_hot(y, 3).float() + 0.01 * torch.randn(60, 3)
    assert knn_accuracy(x, y, x, y, num_classes=3, k=5) == 1.0


def test_linear_eval_keeps_encoder_frozen_and_reports_metrics(make_cfg):
    train_cfg = make_cfg()
    model = build_model(train_cfg).eval()
    before = state_checksum(model.encoder)
    lcfg = LinearEvalConfig(checkpoint="unused", dataset="synthetic", train_subset_size=200,
                            test_subset_size=100, epochs=5, batch_size=64, num_workers=0, device="cpu",
                            val_fraction=0.2)
    res = run_linear_evaluation(model, lcfg, train_cfg, torch.device("cpu"))
    assert state_checksum(model.encoder) == before and res["encoder_frozen_verified"]
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert res["encoder_feature_dim"] == 512
    assert res["num_labeled_train_samples_used_to_fit"] == 160 and res["num_validation_samples"] == 40
    assert res["num_test_samples"] == 100
    assert 0.0 <= res["test_top1_accuracy"] <= 1.0
    assert len(res["test_confusion_matrix"]) == 10
