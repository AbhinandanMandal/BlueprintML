"""Reusable Manim building blocks shared by every scene."""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    DecimalMatrix,
    MathTex,
    RoundedRectangle,
    Square,
    Text,
    VGroup,
    interpolate_color,
    ManimColor,
)

FONT = "Sans"
TOKEN_COLORS = ["#4FC3F7", "#FFB74D", "#81C784", "#F06292"]
Q_COLOR = "#64B5F6"
K_COLOR = "#FFB74D"
V_COLOR = "#81C784"
LOW = ManimColor("#1A2333")
HIGH = ManimColor("#F9D65C")
NEG = ManimColor("#5C8DF9")


def label(text: str, size: float = 28, color=WHITE) -> Text:
    return Text(text, font=FONT, font_size=size, color=color)


def caption(text: str, size: float = 26) -> Text:
    """Subtitle-style text pinned to the bottom of the frame."""
    text_mob = label(text, size=size, color=GREY_B)
    if text_mob.width > 13.0:
        text_mob.scale_to_fit_width(13.0)
    return text_mob.to_edge(DOWN, buff=0.35)


def token_chip(token: str, color, size: float = 30) -> VGroup:
    word = label(token, size=size)
    box = RoundedRectangle(
        corner_radius=0.12,
        width=max(word.width + 0.4, 1.1),
        height=word.height + 0.35,
        stroke_color=color,
        fill_color=color,
        fill_opacity=0.18,
    )
    return VGroup(box, word.move_to(box))


def token_row(tokens: list[str], buff: float = 0.35) -> VGroup:
    chips = VGroup(
        *[token_chip(t, TOKEN_COLORS[i % len(TOKEN_COLORS)]) for i, t in enumerate(tokens)]
    )
    return chips.arrange(RIGHT, buff=buff)


def number_matrix(values: np.ndarray, decimals: int = 2, scale: float = 0.55, color=WHITE) -> DecimalMatrix:
    m = DecimalMatrix(
        np.round(values, decimals),
        element_to_mobject_config={"num_decimal_places": decimals},
        h_buff=1.45,
        v_buff=0.8,
    )
    m.scale(scale)
    m.get_entries().set_color(color)
    m.get_brackets().set_color(GREY_B)
    return m


def named_matrix(name_tex: str, values: np.ndarray, color=WHITE, scale: float = 0.55, shape_note: str | None = None) -> VGroup:
    """`name = [matrix]` with an optional shape note underneath."""
    name = MathTex(name_tex, color=color).scale(0.8)
    eq = MathTex("=").scale(0.8)
    mat = number_matrix(values, scale=scale, color=color)
    group = VGroup(name, eq, mat).arrange(RIGHT, buff=0.15)
    if shape_note:
        note = MathTex(shape_note, color=GREY_B).scale(0.5).next_to(mat, DOWN, buff=0.12)
        group.add(note)
    return group


def heat_color(value: float, vmin: float, vmax: float, diverging: bool = False):
    if diverging:
        bound = max(abs(vmin), abs(vmax)) or 1.0
        t = value / bound
        return interpolate_color(LOW, HIGH, t) if t >= 0 else interpolate_color(LOW, NEG, -t)
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    return interpolate_color(LOW, HIGH, float(np.clip(t, 0, 1)))


class Heatmap(VGroup):
    """Grid of coloured cells with printed values and token labels.

    cells[i][j] is the square for row i, column j; values[i][j] its number.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        cell: float = 0.85,
        decimals: int = 2,
        vmin: float | None = None,
        vmax: float | None = None,
        diverging: bool = False,
        font_size: float = 22,
    ):
        super().__init__()
        self.matrix = matrix
        vmin = float(matrix.min()) if vmin is None else vmin
        vmax = float(matrix.max()) if vmax is None else vmax
        n_rows, n_cols = matrix.shape
        self.cells = VGroup()
        self.values = VGroup()
        for i in range(n_rows):
            row_cells, row_vals = VGroup(), VGroup()
            for j in range(n_cols):
                v = float(matrix[i, j])
                sq = Square(side_length=cell, stroke_width=1.5, stroke_color=GREY_B)
                sq.set_fill(heat_color(v, vmin, vmax, diverging), opacity=1)
                sq.move_to(np.array([j * cell, -i * cell, 0]))
                r, g, b = heat_color(v, vmin, vmax, diverging).to_rgb()
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "#111111" if luminance > 0.5 else WHITE
                num = label(f"{v:.{decimals}f}", size=font_size, color=txt_color).move_to(sq)
                row_cells.add(sq)
                row_vals.add(num)
            self.cells.add(row_cells)
            self.values.add(row_vals)
        self.row_labels = VGroup(
            *[
                label(t, size=font_size + 2, color=TOKEN_COLORS[i % 4]).next_to(self.cells[i][0], LEFT, buff=0.2)
                for i, t in enumerate(row_labels)
            ]
        )
        self.col_labels = VGroup(
            *[
                label(t, size=font_size + 2, color=TOKEN_COLORS[j % 4]).next_to(self.cells[0][j], UP, buff=0.15)
                for j, t in enumerate(col_labels)
            ]
        )
        self.add(self.cells, self.values, self.row_labels, self.col_labels)

    def row(self, i: int) -> VGroup:
        return VGroup(self.cells[i], self.values[i])

    def row_box(self, i: int, color=YELLOW) -> RoundedRectangle:
        r = self.cells[i]
        box = RoundedRectangle(
            corner_radius=0.08, width=r.width + 0.12, height=r.height + 0.12, stroke_color=color, stroke_width=4
        )
        return box.move_to(r)


def vector_column(values: np.ndarray, color, cell: float = 0.42) -> VGroup:
    """A vector drawn as a column of cells shaded by sign and magnitude."""
    col = VGroup()
    bound = max(np.abs(values).max(), 1e-9)
    for idx, v in enumerate(values):
        sq = Square(side_length=cell, stroke_width=1.2, stroke_color=color)
        sq.set_fill(color, opacity=0.15 + 0.75 * abs(v) / bound)
        sq.move_to(np.array([0, -idx * cell, 0]))
        col.add(sq)
    return col


def bar_chart(values: np.ndarray, names: list[str], max_height: float = 2.4, width: float = 0.6, color=BLUE_E, decimals: int = 2) -> VGroup:
    """Minimal vertical bar chart with value labels above the bars."""
    from manim import Rectangle

    vmax = max(values.max(), 1e-9)
    group = VGroup()
    for i, (v, n) in enumerate(zip(values, names)):
        h = max_height * v / vmax
        bar = Rectangle(width=width, height=max(h, 0.02), stroke_width=0, fill_color=color, fill_opacity=0.9)
        bar.move_to(np.array([i * (width + 0.35), h / 2, 0]))
        val = label(f"{v:.{decimals}f}", size=20).next_to(bar, UP, buff=0.08)
        name = label(n, size=22, color=TOKEN_COLORS[i % 4]).next_to(bar, DOWN, buff=0.12)
        name.set_y(-0.25)
        group.add(VGroup(bar, val, name))
    return group
