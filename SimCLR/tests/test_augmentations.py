import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from src.simclr.augmentations import (TwoViewTransform, build_eval_transform, build_train_transform,
                                  visualize_pairs)
from src.simclr.config import TrainConfig
from src.simclr.datasets import (TransformDataset, build_pretrain_dataset, build_subset,
                             deterministic_subset_indices, load_base_dataset)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


def test_two_views_are_independent_and_valid(image):
    tf = TwoViewTransform(build_train_transform())
    torch.manual_seed(0)
    v1, v2 = tf(image)
    assert v1.shape == v2.shape == (3, 32, 32)
    assert v1.dtype == v2.dtype == torch.float32
    assert not torch.allclose(v1, v2)
    # many draws: views practically never coincide
    assert all(not torch.allclose(*tf(image)) for _ in range(20))


def test_training_transform_is_stochastic_but_seed_reproducible(image):
    tf = build_train_transform()
    torch.manual_seed(1)
    a = tf(image)
    torch.manual_seed(1)
    b = tf(image)
    assert torch.equal(a, b)
    assert not torch.equal(tf(image), tf(image))


def test_eval_transform_is_deterministic(image):
    tf = build_eval_transform()
    assert torch.equal(tf(image), tf(image))


def test_upsampled_resolution_option(image):
    assert build_train_transform(image_size=64)(image).shape == (3, 64, 64)
    assert build_eval_transform(image_size=64)(image).shape == (3, 64, 64)


def test_blur_option_runs(image):
    assert build_train_transform(blur=True)(image).shape == (3, 32, 32)


def test_pretrain_dataset_yields_only_two_views_no_labels(make_cfg):
    ds = build_pretrain_dataset(make_cfg())
    item = ds[0]
    assert isinstance(item, tuple) and len(item) == 2
    assert all(isinstance(v, torch.Tensor) and v.shape == (3, 32, 32) for v in item)
    assert ds.return_labels is False


def test_subset_is_deterministic_and_seed_dependent():
    a = deterministic_subset_indices(1000, 100, seed=3)
    assert np.array_equal(a, deterministic_subset_indices(1000, 100, seed=3))
    assert not np.array_equal(a, deterministic_subset_indices(1000, 100, seed=4))
    assert len(set(a.tolist())) == 100
    assert np.array_equal(deterministic_subset_indices(10, 0, seed=0), np.arange(10))
    with pytest.raises(ValueError):
        deterministic_subset_indices(10, 11, seed=0)


def _digests(subset):
    return {hashlib.md5(np.asarray(img).tobytes()).hexdigest() for img, _ in (subset[i] for i in range(len(subset)))}


def test_train_and_test_splits_do_not_overlap(tmp_path):
    train = build_subset("synthetic", str(tmp_path), True, 200, seed=0)
    test = build_subset("synthetic", str(tmp_path), False, 200, seed=1)
    assert not _digests(train) & _digests(test)


def test_cifar10_loads_if_available():
    """Not runnable offline. Runs only when CIFAR-10 is already on disk at ./data."""
    try:
        ds = load_base_dataset("cifar10", "./data", train=True, download=False)
    except RuntimeError:
        pytest.skip("CIFAR-10 not available locally")
    img, label = ds[0]
    assert img.size == (32, 32) and 0 <= label < 10 and len(ds) == 50000


def test_visualize_pairs_writes_file(make_cfg, tmp_path):
    pytest.importorskip("matplotlib")
    out = visualize_pairs(build_pretrain_dataset(make_cfg()), str(tmp_path / "pairs.png"), num_images=3)
    assert out.exists() and out.stat().st_size > 0
