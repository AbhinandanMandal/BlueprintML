"""Scenes for "What Does an AI Model Actually Pay Attention To?".

Each part of the video is a function `part_*(scene, model)` that draws,
animates, and then clears the frame. Every part also has its own Scene
class, so it can be rendered on its own while iterating. `FullExplainer`
plays all parts in order.

Render from the AttentionExplained/ folder, for example:

    manim -pql src/attention_viz/scenes.py FullExplainer
    manim -pql src/attention_viz/scenes.py SoftmaxScene
"""

from __future__ import annotations

import sys
from pathlib import Path

# Manim loads this file as a standalone module, so make the package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from manim import (  # noqa: E402
    DOWN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    UR,
    YELLOW,
    Arrow,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    LaggedStart,
    MathTex,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
)

from attention_viz.attention_math import (  # noqa: E402
    D_K,
    D_MODEL,
    ToyTransformer,
    build_toy_model,
    project_2d,
)
from attention_viz.components import (  # noqa: E402
    K_COLOR,
    Q_COLOR,
    TOKEN_COLORS,
    V_COLOR,
    Heatmap,
    bar_chart,
    caption,
    label,
    named_matrix,
    number_matrix,
    token_row,
    vector_column,
)

QUERY = 2  # index of "sat", the query followed through the video


def clear(scene: Scene, run_time: float = 0.8) -> None:
    if scene.mobjects:
        scene.play(*[FadeOut(m) for m in scene.mobjects], run_time=run_time)


def swap_caption(scene: Scene, old: Text | None, text: str) -> Text:
    new = caption(text)
    if old is None:
        scene.play(FadeIn(new, shift=UP * 0.2), run_time=0.6)
    else:
        scene.play(ReplacementTransform(old, new), run_time=0.6)
    return new


def title(text: str) -> Text:
    return label(text, size=36).to_edge(UP, buff=0.4)


# --------------------------------------------------------------------------
# Part 0: title
# --------------------------------------------------------------------------
def part_intro(scene: Scene, model: ToyTransformer) -> None:
    head = label("What does an AI model actually pay attention to?", size=40)
    sub = label("Scaled dot-product self-attention, one step at a time", size=28, color=GREY_B)
    VGroup(head, sub).arrange(DOWN, buff=0.35).shift(UP * 1.4)
    eq = MathTex(
        r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"
    ).scale(1.0)
    src = label("Vaswani et al., 2017, Attention Is All You Need, Eq. 1", size=22, color=GREY_B)
    VGroup(eq, src).arrange(DOWN, buff=0.3).shift(DOWN * 0.6)
    note = caption(
        f"Toy model: {len(model.tokens)} tokens, d_model = {D_MODEL}, d_k = d_v = {D_K}, random untrained weights"
    )
    scene.play(Write(head), run_time=1.5)
    scene.play(FadeIn(sub))
    scene.play(Write(eq), run_time=2)
    scene.play(FadeIn(src), FadeIn(note))
    scene.wait(2.5)
    clear(scene)


# --------------------------------------------------------------------------
# Part 1: tokens become vectors
# --------------------------------------------------------------------------
def part_embeddings(scene: Scene, model: ToyTransformer) -> None:
    head = title("1. Tokens become vectors")
    tokens = token_row(model.tokens).next_to(head, DOWN, buff=0.5)
    scene.play(FadeIn(head), LaggedStart(*[FadeIn(t, shift=DOWN * 0.2) for t in tokens], lag_ratio=0.2))
    cap = swap_caption(scene, None, "Each token is looked up in an embedding table. Here the table is random.")

    e = number_matrix(model.embeddings, scale=0.5)
    pe = number_matrix(model.positional, scale=0.5)
    x = number_matrix(model.x0, scale=0.5)
    plus, eq = MathTex("+"), MathTex("=")
    row = VGroup(e, plus, pe, eq, x).arrange(RIGHT, buff=0.3)
    row_tags = VGroup()
    for i, t in enumerate(model.tokens):
        tag = label(t, size=22, color=TOKEN_COLORS[i])
        tag.set_y(e.get_rows()[i].get_y())
        tag.set_x(e.get_left()[0] - 0.2 - tag.width / 2)
        row_tags.add(tag)
    VGroup(row_tags, row).move_to(DOWN * 0.4)
    names = VGroup(
        MathTex("E").next_to(e, UP, buff=0.15),
        MathTex(r"PE").next_to(pe, UP, buff=0.15),
        MathTex("X").next_to(x, UP, buff=0.15),
    )

    # Each chip's word slides down into its row of E.
    scene.play(*[TransformFromCopy(tokens[i][1], row_tags[i]) for i in range(len(tokens))], run_time=1.2)
    scene.play(FadeIn(e), FadeIn(names[0]))
    scene.wait(1)
    cap = swap_caption(scene, cap, "Attention ignores order, so a sinusoidal positional encoding is added (Sec. 3.5).")
    scene.play(FadeIn(plus), FadeIn(pe), FadeIn(names[1]))
    pe_eq = MathTex(
        r"PE_{(pos,2i)} = \sin\!\left(pos / 10000^{2i/d_{model}}\right)", color=GREY_B
    ).scale(0.6).next_to(row, DOWN, buff=0.35)
    scene.play(Write(pe_eq))
    scene.wait(1.5)
    scene.play(FadeIn(eq), TransformFromCopy(VGroup(e, pe), x), FadeIn(names[2]))
    cap = swap_caption(scene, cap, "Row i of X is the input vector for token i. Everything below starts from X.")
    scene.wait(2.5)
    clear(scene)


# --------------------------------------------------------------------------
# Part 2: Q, K, V projections
# --------------------------------------------------------------------------
def part_qkv(scene: Scene, model: ToyTransformer) -> None:
    layer = model.layers[0]
    head = title("2. Three projections of the same input")
    scene.play(FadeIn(head))
    eqs = MathTex(
        r"Q = XW^{Q}", r"\quad K = XW^{K}", r"\quad V = XW^{V}"
    ).scale(0.9).next_to(head, DOWN, buff=0.35)
    eqs[0].set_color(Q_COLOR)
    eqs[1].set_color(K_COLOR)
    eqs[2].set_color(V_COLOR)
    shapes = MathTex(
        r"(4\times4)\,(4\times3) \rightarrow (4\times3)", color=GREY_B
    ).scale(0.6).next_to(eqs, DOWN, buff=0.15)
    scene.play(Write(eqs), run_time=1.5)
    scene.play(FadeIn(shapes))
    cap = swap_caption(scene, None, "W^Q, W^K and W^V are learned in a real model. Here they are random.")

    mats = VGroup(
        named_matrix("Q", layer.q, color=Q_COLOR, scale=0.5),
        named_matrix("K", layer.k, color=K_COLOR, scale=0.5),
        named_matrix("V", layer.v, color=V_COLOR, scale=0.5),
    ).arrange(RIGHT, buff=0.7).shift(DOWN * 0.6)
    for m in mats:
        scene.play(FadeIn(m, shift=UP * 0.2), run_time=0.8)
    scene.wait(1)
    roles = VGroup(
        label("query: what this position compares against the others", size=22, color=Q_COLOR),
        label("key: what each position is compared on", size=22, color=K_COLOR),
        label("value: what each position hands over", size=22, color=V_COLOR),
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(mats, DOWN, buff=0.35)
    cap = swap_caption(scene, cap, "A common reading of the three roles. It is an intuition, not part of the maths.")
    scene.play(LaggedStart(*[FadeIn(r) for r in roles], lag_ratio=0.3))
    scene.wait(3)
    clear(scene)


# --------------------------------------------------------------------------
# Part 3: scores
# --------------------------------------------------------------------------
def part_scores(scene: Scene, model: ToyTransformer) -> None:
    layer = model.layers[0]
    toks = model.tokens
    head = title("3. Scores: every query against every key")
    eq = MathTex(r"s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}").scale(0.9).next_to(head, DOWN, buff=0.3)
    scene.play(FadeIn(head), Write(eq))

    q = named_matrix("Q", layer.q, color=Q_COLOR, scale=0.55).to_edge(LEFT, buff=0.6).shift(UP * 0.3)
    heat = Heatmap(layer.scores, toks, toks, cell=0.8, diverging=True)
    heat.next_to(eq, DOWN, buff=0.3).to_edge(RIGHT, buff=0.9)
    scene.play(FadeIn(q))
    cap = swap_caption(scene, None, "Follow the query for \"sat\" (row 3) against the key for \"cat\".")

    q_row = SurroundingRectangle(q[2].get_rows()[QUERY], color=Q_COLOR, buff=0.08)
    qi, kj = layer.q[QUERY], layer.k[1]
    raw = float(qi @ kj)
    worked = MathTex(
        r"q_{\text{sat}} \cdot k_{\text{cat}} = ",
        f"{raw:.2f}",
        r"\qquad \frac{" + f"{raw:.2f}" + r"}{\sqrt{3}} = ",
        f"{raw / np.sqrt(D_K):.2f}",
    ).scale(0.65).next_to(q, DOWN, buff=0.4).align_to(q, LEFT)
    scene.play(Create(q_row))
    scene.play(Write(worked), run_time=1.5)
    scene.wait(1)

    scene.play(FadeIn(heat.row_labels), FadeIn(heat.col_labels), FadeIn(heat.cells))
    target = VGroup(heat.cells[QUERY][1], heat.values[QUERY][1])
    scene.play(TransformFromCopy(worked[3], heat.values[QUERY][1]), run_time=1.2)
    scene.play(Create(SurroundingRectangle(target, color=YELLOW, buff=0.02)))
    others = [heat.values[i][j] for i in range(len(toks)) for j in range(len(toks)) if (i, j) != (QUERY, 1)]
    scene.play(LaggedStart(*[FadeIn(v) for v in others], lag_ratio=0.05), run_time=2)
    cap = swap_caption(
        scene, cap, "Dividing by the square root of d_k keeps scores from growing with dimension (Sec. 3.2.1)."
    )
    scene.wait(3)
    clear(scene)


# --------------------------------------------------------------------------
# Part 4: softmax
# --------------------------------------------------------------------------
def part_softmax(scene: Scene, model: ToyTransformer) -> None:
    layer = model.layers[0]
    toks = model.tokens
    head = title("4. Softmax turns each row into weights")
    eq = MathTex(
        r"\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k}\exp(s_{ik})}"
    ).scale(0.85).next_to(head, DOWN, buff=0.3)
    scene.play(FadeIn(head), Write(eq))

    scores = Heatmap(layer.scores, toks, toks, cell=0.78, diverging=True)
    weights = Heatmap(layer.weights, toks, toks, cell=0.78, vmin=0.0, vmax=1.0)
    scores.next_to(eq, DOWN, buff=0.3).set_x(-4.0)
    weights.next_to(eq, DOWN, buff=0.3).set_x(2.6)
    s_tag = label("scores s", size=22, color=GREY_B).next_to(scores.cells, DOWN, buff=0.2)
    w_tag = label("attention weights A", size=22, color=GREY_B).next_to(weights.cells, DOWN, buff=0.2)
    arrow = Arrow(
        scores.cells.get_right() + RIGHT * 0.1, weights.row_labels.get_left() + LEFT * 0.1, buff=0.1
    ).set_y(scores.cells.get_y())
    arrow_tag = label("softmax\nper row", size=20).next_to(arrow, UP, buff=0.1)

    scene.play(FadeIn(scores), FadeIn(s_tag))
    cap = swap_caption(scene, None, "Softmax runs across each row, one query at a time.")
    scene.play(Create(arrow), FadeIn(arrow_tag))
    scene.play(FadeIn(weights.row_labels), FadeIn(weights.col_labels), FadeIn(w_tag))
    for i in range(len(toks)):
        scene.play(TransformFromCopy(scores.row(i), weights.row(i)), run_time=0.7)

    sums = VGroup(
        *[
            label(f"= {layer.weights[i].sum():.2f}", size=22, color=GREY_B).next_to(weights.cells[i][-1], RIGHT, buff=0.15)
            for i in range(len(toks))
        ]
    )
    cap = swap_caption(scene, cap, "Each row is non-negative and sums to 1: a distribution over the keys.")
    scene.play(LaggedStart(*[FadeIn(s) for s in sums], lag_ratio=0.2))
    box = weights.row_box(QUERY)
    scene.play(Create(box))
    top = int(np.argmax(layer.weights[QUERY]))
    cap = swap_caption(
        scene, cap, f"\"sat\" puts its largest weight, {layer.weights[QUERY, top]:.2f}, on \"{toks[top]}\"."
    )
    scene.wait(3)
    clear(scene)


# --------------------------------------------------------------------------
# Part 5: weighted sum of values
# --------------------------------------------------------------------------
def part_weighted_values(scene: Scene, model: ToyTransformer) -> None:
    layer = model.layers[0]
    toks = model.tokens
    alpha = layer.weights[QUERY]
    head = title("5. The output is a weighted sum of values")
    eq = MathTex(
        r"z_{\text{sat}} = \sum_{j} \alpha_{\text{sat},j}\, v_j", r"\qquad Z = AV"
    ).scale(0.85).next_to(head, DOWN, buff=0.3)
    scene.play(FadeIn(head), Write(eq))

    cols = VGroup()
    for j, t in enumerate(toks):
        col = vector_column(layer.v[j], V_COLOR, cell=0.55)
        name = label(t, size=22, color=TOKEN_COLORS[j]).next_to(col, UP, buff=0.15)
        vlabel = MathTex(f"v_{{{j + 1}}}", color=V_COLOR).scale(0.6).next_to(name, UP, buff=0.08)
        a = MathTex(r"\times\," + f"{alpha[j]:.2f}").scale(0.6).next_to(col, DOWN, buff=0.2)
        cols.add(VGroup(col, name, vlabel, a))
    cols.arrange(RIGHT, buff=0.75).to_edge(LEFT, buff=0.9).shift(DOWN * 0.6)
    scene.play(LaggedStart(*[FadeIn(c[:3]) for c in cols], lag_ratio=0.2))
    cap = swap_caption(scene, None, "Each value vector is scaled by how much \"sat\" attends to it.")
    scene.play(LaggedStart(*[Write(c[3]) for c in cols], lag_ratio=0.2))
    scene.play(*[c[0].animate.set_opacity(0.15 + 0.85 * alpha[j] / alpha.max()) for j, c in enumerate(cols)])
    scene.wait(1)

    z_col = vector_column(layer.z[QUERY], YELLOW, cell=0.55)
    z_nums = number_matrix(layer.z[QUERY][:, None], scale=0.6, color=YELLOW)
    z_group = VGroup(z_col, z_nums).arrange(RIGHT, buff=0.35)
    z_group.next_to(cols, RIGHT, buff=1.6).align_to(cols[0][0], UP)
    z_name = MathTex(r"z_{\text{sat}}", color=YELLOW).scale(0.8).next_to(z_group, UP, buff=0.2)
    equals = MathTex("=").scale(1.2).move_to(VGroup(cols[-1][0], z_col)).set_y(z_col.get_y())
    # All four scaled columns slide into the output slot, then become z.
    copies = [c[0].copy() for c in cols]
    scene.play(*[cp.animate.move_to(z_col) for cp in copies], FadeIn(equals), run_time=1.6)
    scene.play(*[FadeOut(cp) for cp in copies], FadeIn(z_col), run_time=0.6)
    scene.play(FadeIn(z_nums), FadeIn(z_name))
    cap = swap_caption(scene, cap, "The same sum runs for every row at once: Z = AV gives a new vector per token.")
    scene.wait(3)
    clear(scene)


# --------------------------------------------------------------------------
# Part 6: across layers
# --------------------------------------------------------------------------
def part_layers(scene: Scene, model: ToyTransformer) -> None:
    toks = model.tokens
    reps = model.representations()
    pts = project_2d(reps)
    bound = max(np.abs(np.concatenate(pts)).max(), 1e-6) * 1.15

    head = title("6. Stacking layers")
    eq = MathTex(
        r"X_{\ell+1} = \mathrm{LayerNorm}\!\left(X_\ell + \mathrm{Attention}(X_\ell)\,W^{O}\right)"
    ).scale(0.7).next_to(head, DOWN, buff=0.25)
    omit = label("(feed-forward sublayer and multiple heads omitted in this toy)", size=20, color=GREY_B).next_to(eq, DOWN, buff=0.1)
    scene.play(FadeIn(head), Write(eq), FadeIn(omit))

    axes = Axes(
        x_range=[-bound, bound, bound],
        y_range=[-bound, bound, bound],
        x_length=3.7,
        y_length=3.7,
        tips=False,
        axis_config={"stroke_color": GREY_B, "stroke_width": 1.5, "include_ticks": False},
    )
    axes.next_to(omit, DOWN, buff=0.3).set_x(-3.6)

    def dots_for(layer_idx: int) -> VGroup:
        group = VGroup()
        for i, t in enumerate(toks):
            p = axes.c2p(*pts[layer_idx][i])
            d = Dot(p, radius=0.11, color=TOKEN_COLORS[i])
            tag = label(t, size=20, color=TOKEN_COLORS[i]).next_to(d, UR, buff=0.05)
            group.add(VGroup(d, tag))
        return group

    dots = dots_for(0)
    stage = label("Layer 0 (input X)", size=26).next_to(axes, DOWN, buff=0.15)
    ax_note = label("2D PCA of token vectors, one basis for all layers", size=18, color=GREY_B)
    ax_note.next_to(stage, DOWN, buff=0.08)
    scene.play(Create(axes), FadeIn(ax_note), FadeIn(stage), FadeIn(dots))
    cap = swap_caption(scene, None, "Every layer rewrites each token as a mix of itself and the others.")

    heat = None
    for ell, layer in enumerate(model.layers, start=1):
        new_heat = Heatmap(layer.weights, toks, toks, cell=0.62, vmin=0.0, vmax=1.0, font_size=18)
        new_heat.move_to(axes).set_x(3.2)
        new_tag = label(f"attention weights, layer {ell}", size=20, color=GREY_B).next_to(new_heat, DOWN, buff=0.2)
        new_stage = label(f"Layer {ell}", size=26).move_to(stage)
        anims = [Transform(dots, dots_for(ell)), Transform(stage, new_stage)]
        if heat is None:
            heat = VGroup(new_heat, new_tag)
            anims.append(FadeIn(heat))
        else:
            anims.append(Transform(heat, VGroup(new_heat, new_tag)))
        scene.play(*anims, run_time=1.8)
        scene.wait(1.2)
    cap = swap_caption(
        scene, cap, "With random weights the motion means nothing linguistic. It shows the mechanism, not a learned pattern."
    )
    scene.wait(1.5)
    cap = swap_caption(
        scene, cap, "After layer 1, position \"sat\" no longer holds only \"sat\". Later heatmaps weight mixtures, not words."
    )
    scene.wait(3)
    clear(scene)


# --------------------------------------------------------------------------
# Part 7: what attention weights cannot establish
# --------------------------------------------------------------------------
def part_limitations(scene: Scene, model: ToyTransformer) -> None:
    layer = model.layers[0]
    toks = model.tokens
    head = title("7. What attention weights do not tell you")
    scene.play(FadeIn(head))

    w = layer.weights[QUERY]
    contrib = layer.weighted_value_norms(QUERY)
    left = bar_chart(w, toks, max_height=2.3, color="#5C8DF9")
    right = bar_chart(contrib, toks, max_height=2.3, color=V_COLOR)
    left_t = MathTex(r"\alpha_{\text{sat},j}").scale(0.8)
    right_t = MathTex(r"\lVert \alpha_{\text{sat},j}\, v_j \rVert").scale(0.8)
    lg = VGroup(left_t, left).arrange(DOWN, buff=0.35)
    rg = VGroup(right_t, right).arrange(DOWN, buff=0.35)
    VGroup(lg, rg).arrange(RIGHT, buff=2.2).shift(DOWN * 0.3)
    scene.play(FadeIn(lg))
    cap = swap_caption(scene, None, f"By weight, \"sat\" attends most to \"{toks[int(np.argmax(w))]}\".")
    scene.play(Create(SurroundingRectangle(left[int(np.argmax(w))], color=YELLOW, buff=0.06)))
    scene.wait(1.5)
    scene.play(FadeIn(rg))
    cap = swap_caption(
        scene,
        cap,
        f"Weighted by value size, \"{toks[int(np.argmax(contrib))]}\" ranks first: it moves the output most.",
    )
    scene.play(Create(SurroundingRectangle(right[int(np.argmax(contrib))], color=YELLOW, buff=0.06)))
    cite = label("Norm-based analysis: Kobayashi et al., 2020", size=20, color=GREY_B).next_to(VGroup(lg, rg), DOWN, buff=0.3)
    scene.play(FadeIn(cite))
    scene.wait(3)
    clear(scene)

    head = title("7. What attention weights do not tell you")
    points = [
        ("A weight ignores the size of the vector it scales.", "Kobayashi et al., 2020"),
        ("Very different weights can give the same prediction,", "Jain & Wallace, 2019"),
        ("though whether that undermines them is disputed.", "Wiegreffe & Pinter, 2019"),
        ("Residual paths carry information around attention.", "Abnar & Zuidema, 2020"),
        ("Deep positions hold mixtures, not single words.", "Brunner et al., 2020"),
        ("One heatmap is one head in one layer of many.", ""),
    ]
    rows = VGroup()
    for text, ref in points:
        t = label(text, size=24)
        r = label(ref, size=18, color=GREY_B)
        rows.add(VGroup(t, r).arrange(RIGHT, buff=0.3, aligned_edge=DOWN))
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).next_to(head, DOWN, buff=0.5)
    scene.play(FadeIn(head))
    for r in rows:
        scene.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.7)
        scene.wait(0.6)
    takeaway = label(
        "Attention weights show where one step reads from. Why a model decides needs tests,",
        size=24,
        color=YELLOW,
    )
    takeaway2 = label("such as removing or changing inputs and measuring the effect.", size=24, color=YELLOW)
    VGroup(takeaway, takeaway2).arrange(DOWN, buff=0.12).next_to(rows, DOWN, buff=0.5)
    scene.play(FadeIn(takeaway), FadeIn(takeaway2))
    cap = swap_caption(scene, None, "Overview of the debate: Bibal et al., 2022")
    scene.wait(4)
    clear(scene)


# --------------------------------------------------------------------------
# Part 8: references
# --------------------------------------------------------------------------
REFERENCES = [
    "Vaswani et al. (2017). Attention Is All You Need. NeurIPS.",
    "Jain & Wallace (2019). Attention is not Explanation. NAACL.",
    "Wiegreffe & Pinter (2019). Attention is not not Explanation. EMNLP.",
    "Brunner et al. (2020). On Identifiability in Transformers. ICLR.",
    "Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers. ACL.",
    "Kobayashi et al. (2020). Attention is Not Only a Weight. EMNLP.",
    "Bibal et al. (2022). Is Attention Explanation? An Introduction to the Debate. ACL.",
]


def part_references(scene: Scene, model: ToyTransformer) -> None:
    head = title("References")
    refs = VGroup(*[label(r, size=22) for r in REFERENCES]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    refs.next_to(head, DOWN, buff=0.5)
    footer = label("BlueprintML: AttentionExplained", size=20, color=GREY_B).to_edge(DOWN, buff=0.4)
    scene.play(FadeIn(head), LaggedStart(*[FadeIn(r) for r in refs], lag_ratio=0.1), FadeIn(footer))
    scene.wait(4)
    clear(scene)


PARTS = [
    part_intro,
    part_embeddings,
    part_qkv,
    part_scores,
    part_softmax,
    part_weighted_values,
    part_layers,
    part_limitations,
    part_references,
]


class _PartScene(Scene):
    part = staticmethod(part_intro)

    def construct(self) -> None:
        self.part(self, build_toy_model())


class IntroScene(_PartScene):
    part = staticmethod(part_intro)


class EmbeddingScene(_PartScene):
    part = staticmethod(part_embeddings)


class QKVScene(_PartScene):
    part = staticmethod(part_qkv)


class ScoresScene(_PartScene):
    part = staticmethod(part_scores)


class SoftmaxScene(_PartScene):
    part = staticmethod(part_softmax)


class WeightedValuesScene(_PartScene):
    part = staticmethod(part_weighted_values)


class LayersScene(_PartScene):
    part = staticmethod(part_layers)


class LimitationsScene(_PartScene):
    part = staticmethod(part_limitations)


class ReferencesScene(_PartScene):
    part = staticmethod(part_references)


class FullExplainer(Scene):
    def construct(self) -> None:
        model = build_toy_model()
        for part in PARTS:
            part(self, model)
