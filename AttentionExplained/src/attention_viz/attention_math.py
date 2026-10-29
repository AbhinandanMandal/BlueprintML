"""Numbers behind every scene.

All matrices drawn in the animation come from this module, so the video
never shows a value that was not computed. The model is a toy: random
weights, four tokens, d_model = 4. It shows the mechanics of scaled
dot-product self-attention (Vaswani et al., 2017, Section 3.2.1), not the
behaviour of a trained network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOKENS = ["the", "cat", "sat", "down"]
D_MODEL = 4
D_K = 3  # also used as d_v
N_LAYERS = 3
# Seed chosen so the norm-based example in the limitations scene shows a
# ranking flip for the query "sat". See README, "Limitations".
SEED = 7


def sinusoidal_positional_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """PE from Section 3.5 of the paper.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    pe = np.zeros((n_positions, d_model))
    positions = np.arange(n_positions)[:, None]
    div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(positions / div)
    pe[:, 1::2] = np.cos(positions / div)
    return pe


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the row max first)."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm without learned gain or bias."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


@dataclass
class AttentionStep:
    """Every intermediate of one single-head self-attention layer."""

    x: np.ndarray  # (T, d_model) input to the layer
    w_q: np.ndarray  # (d_model, d_k)
    w_k: np.ndarray  # (d_model, d_k)
    w_v: np.ndarray  # (d_model, d_v)
    w_o: np.ndarray  # (d_v, d_model)
    q: np.ndarray = field(init=False)  # (T, d_k)
    k: np.ndarray = field(init=False)  # (T, d_k)
    v: np.ndarray = field(init=False)  # (T, d_v)
    scores: np.ndarray = field(init=False)  # (T, T) = QK^T / sqrt(d_k)
    weights: np.ndarray = field(init=False)  # (T, T) row-wise softmax
    z: np.ndarray = field(init=False)  # (T, d_v) = weights @ V
    out: np.ndarray = field(init=False)  # (T, d_model) = LN(x + z W_O)

    def __post_init__(self) -> None:
        d_k = self.w_q.shape[1]
        self.q = self.x @ self.w_q
        self.k = self.x @ self.w_k
        self.v = self.x @ self.w_v
        self.scores = self.q @ self.k.T / np.sqrt(d_k)
        self.weights = softmax(self.scores, axis=-1)
        self.z = self.weights @ self.v
        # Post-LN residual block, as in the original paper (Section 3.1).
        self.out = layer_norm(self.x + self.z @ self.w_o)

    def weighted_value_norms(self, query: int) -> np.ndarray:
        """||alpha_ij * v_j|| for every key j (Kobayashi et al., 2020)."""
        return self.weights[query] * np.linalg.norm(self.v, axis=1)


@dataclass
class ToyTransformer:
    """A stack of single-head attention layers with fixed random weights.

    No feed-forward sublayer, no multi-head split, no training. The point
    is to make each matrix small enough to print on screen.
    """

    tokens: list[str]
    embeddings: np.ndarray  # (T, d_model) token embeddings before PE
    positional: np.ndarray  # (T, d_model)
    layers: list[AttentionStep]

    @property
    def x0(self) -> np.ndarray:
        return self.embeddings + self.positional

    def representations(self) -> list[np.ndarray]:
        """[X_0, X_1, ..., X_L]: input followed by each layer's output."""
        return [self.x0] + [layer.out for layer in self.layers]


def build_toy_model(
    tokens: list[str] | None = None,
    d_model: int = D_MODEL,
    d_k: int = D_K,
    n_layers: int = N_LAYERS,
    seed: int = SEED,
) -> ToyTransformer:
    tokens = list(tokens or TOKENS)
    rng = np.random.default_rng(seed)
    t = len(tokens)
    embeddings = rng.normal(0.0, 1.0, size=(t, d_model))
    positional = sinusoidal_positional_encoding(t, d_model)
    x = embeddings + positional

    # Scale like a standard init so scores stay in a readable range.
    scale_in = 1.0 / np.sqrt(d_model)
    scale_out = 1.0 / np.sqrt(d_k)
    layers = []
    for _ in range(n_layers):
        step = AttentionStep(
            x=x,
            w_q=rng.normal(0.0, scale_in, size=(d_model, d_k)),
            w_k=rng.normal(0.0, scale_in, size=(d_model, d_k)),
            w_v=rng.normal(0.0, scale_in, size=(d_model, d_k)),
            w_o=rng.normal(0.0, scale_out, size=(d_k, d_model)),
        )
        layers.append(step)
        x = step.out
    return ToyTransformer(tokens, embeddings, positional, layers)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    normed = x / np.linalg.norm(x, axis=1, keepdims=True)
    return normed @ normed.T


def project_2d(reps: list[np.ndarray]) -> list[np.ndarray]:
    """Project every layer's token vectors onto one shared 2D PCA basis.

    One basis for all layers, so movement between frames reflects change in
    the representations rather than a change of axes.
    """
    stacked = np.concatenate(reps, axis=0)
    centred = stacked - stacked.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:2].T
    mean = stacked.mean(axis=0)
    return [(r - mean) @ basis for r in reps]


def ranking_flips(step: AttentionStep, query: int) -> bool:
    """True if the top key by weight differs from the top key by ||alpha v||."""
    by_weight = int(np.argmax(step.weights[query]))
    by_norm = int(np.argmax(step.weighted_value_norms(query)))
    return by_weight != by_norm
