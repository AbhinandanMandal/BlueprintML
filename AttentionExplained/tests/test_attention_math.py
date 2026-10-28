"""Checks that the numbers drawn in the video are correct.

Run from AttentionExplained/:  python -m pytest tests -q
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_viz.attention_math import (  # noqa: E402
    AttentionStep,
    build_toy_model,
    project_2d,
    ranking_flips,
    sinusoidal_positional_encoding,
    softmax,
)


@pytest.fixture(scope="module")
def model():
    return build_toy_model()


def test_shapes(model):
    t = len(model.tokens)
    layer = model.layers[0]
    assert model.x0.shape == (t, 4)
    assert layer.q.shape == layer.k.shape == layer.v.shape == (t, 3)
    assert layer.scores.shape == layer.weights.shape == (t, t)
    assert layer.out.shape == (t, 4)


def test_softmax_rows_are_distributions(model):
    for layer in model.layers:
        assert np.all(layer.weights >= 0)
        np.testing.assert_allclose(layer.weights.sum(axis=1), 1.0, atol=1e-12)


def test_softmax_is_shift_invariant():
    x = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(softmax(x), softmax(x + 100.0))


def test_matches_paper_equation(model):
    """softmax(QK^T / sqrt(d_k)) V, written out directly."""
    layer = model.layers[0]
    q, k, v = model.x0 @ layer.w_q, model.x0 @ layer.w_k, model.x0 @ layer.w_v
    s = q @ k.T / np.sqrt(k.shape[1])
    a = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(layer.z, a @ v, atol=1e-12)


def test_positional_encoding_matches_formula():
    pe = sinusoidal_positional_encoding(4, 4)
    np.testing.assert_allclose(pe[0], [0.0, 1.0, 0.0, 1.0])
    pos, i, d = 3, 1, 4
    np.testing.assert_allclose(pe[pos, 2 * i], np.sin(pos / 10000 ** (2 * i / d)))
    np.testing.assert_allclose(pe[pos, 2 * i + 1], np.cos(pos / 10000 ** (2 * i / d)))


def test_layers_chain(model):
    for prev, nxt in zip(model.layers, model.layers[1:]):
        np.testing.assert_array_equal(prev.out, nxt.x)


def test_limitations_example_holds(model):
    """The video says "sat" weights "the" most but "cat" contributes most."""
    layer = model.layers[0]
    sat = model.tokens.index("sat")
    assert model.tokens[int(np.argmax(layer.weights[sat]))] == "the"
    assert model.tokens[int(np.argmax(layer.weighted_value_norms(sat)))] == "cat"
    assert ranking_flips(layer, sat)


def test_projection_uses_one_basis(model):
    pts = project_2d(model.representations())
    assert len(pts) == len(model.layers) + 1
    assert all(p.shape == (len(model.tokens), 2) for p in pts)


def test_agrees_with_repo_multihead_attention():
    """Single-head case of DecoderOnlyTransformer/transformer/attention.py."""
    torch = pytest.importorskip("torch")
    path = ROOT.parent / "DecoderOnlyTransformer" / "transformer" / "attention.py"
    if not path.exists():
        pytest.skip("DecoderOnlyTransformer not present")
    spec = importlib.util.spec_from_file_location("repo_attention", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # The repo's class ties d_k to d_model // h, so use d_k = d_model here.
    toy = build_toy_model(d_k=4, n_layers=1)
    step: AttentionStep = toy.layers[0]
    mha = mod.MultiHeadAttention(d_model=4, h=1, dropout=0.0).double().eval()
    with torch.no_grad():
        mha.W_Q.weight.copy_(torch.from_numpy(step.w_q.T))
        mha.W_K.weight.copy_(torch.from_numpy(step.w_k.T))
        mha.W_V.weight.copy_(torch.from_numpy(step.w_v.T))
        mha.W_O.weight.copy_(torch.from_numpy(step.w_o.T))
        mha.W_O.bias.zero_()
        x = torch.from_numpy(step.x)[None]
        out = mha(x, x, x)[0].numpy()
    np.testing.assert_allclose(out, step.z @ step.w_o, atol=1e-10)
