import torch

from moco_reproduction.config import MoCoConfig
from moco_reproduction.datasets import build_data, make_ssl_loader


def cfg(**kw):
    base = dict(dataset="synthetic", subset_size=64, batch_size=16, queue_size=64,
                eval_train_size=100, eval_test_size=50, num_workers=0)
    base.update(kw)
    return MoCoConfig(**base).validate()


def test_two_views_shapes_finite_and_no_labels():
    c = cfg()
    d = build_data(c)
    assert len(d.ssl_dataset) == 64
    item = d.ssl_dataset[0]
    assert len(item) == 2 and item[0].shape == (3, 32, 32) and item[1].shape == (3, 32, 32)
    xq, xk = next(iter(make_ssl_loader(d.ssl_dataset, c)))
    assert xq.shape == (16, 3, 32, 32) and xk.shape == (16, 3, 32, 32)
    assert torch.isfinite(xq).all() and torch.isfinite(xk).all()
    assert not torch.equal(xq, xk)          # independently sampled views


def test_subset_is_fixed_for_a_seed_and_differs_across_seeds():
    a, b = build_data(cfg(seed=42)).ssl_indices, build_data(cfg(seed=42)).ssl_indices
    c = build_data(cfg(seed=7)).ssl_indices
    assert a == b and a != c and len(set(a)) == len(a)
