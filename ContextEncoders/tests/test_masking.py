import numpy as np
import pytest
import torch

from src.context_encoders.masking import make_mask, mask_block, mask_central, mask_region

H = W = 128


def test_central_mask_geometry():
    m = mask_central(None, 3)
    assert m.shape == (3, H, W) and m.dtype == np.float32
    assert m.sum(axis=(1, 2)).tolist() == [64 * 64] * 3
    assert m[0, 32:96, 32:96].all() and m[0].sum() == m[0, 32:96, 32:96].sum()


@pytest.mark.parametrize("fn", [mask_block, mask_region])
def test_random_masks_binary_capped_and_seeded(fn):
    a = fn(np.random.default_rng(1), 4)
    b = fn(np.random.default_rng(1), 4)
    assert a.shape == (4, H, W)
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert (a.sum(axis=(1, 2)) <= H * W // 4).all() and (a.sum(axis=(1, 2)) > 0).all()
    np.testing.assert_array_equal(a, b)


def test_make_mask_and_alignment_with_image():
    rng = np.random.default_rng(0)
    img = torch.randn(2, 3, H, W)
    mask = make_mask(rng, 2, "central")
    assert mask.shape == (2, 1, H, W)
    masked = img * (1 - mask)
    assert (masked[:, :, 32:96, 32:96] == 0).all()                  # hole zero-filled
    assert torch.equal(masked[:, :, :32], img[:, :, :32])            # known pixels untouched
    with pytest.raises(ValueError):
        make_mask(rng, 2, "bogus")
