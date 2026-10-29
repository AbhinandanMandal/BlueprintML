# Storyboard and maths

This file pairs each part of the video with the equation it animates and the
numbers it shows. All numbers come from
[`src/attention_viz/attention_math.py`](src/attention_viz/attention_math.py)
with `SEED = 7`.

## Notation

Notation follows Vaswani et al. (2017), Section 3.2.

| Symbol | Meaning | Size in the toy |
|---|---|---|
| $T$ | number of tokens | 4 (`the cat sat down`) |
| $d_{model}$ | width of each token vector | 4 |
| $d_k = d_v$ | width of queries, keys and values | 3 |
| $X$ | input matrix, one row per token | $4 \times 4$ |
| $W^Q, W^K, W^V$ | projection matrices | $4 \times 3$ |
| $W^O$ | output projection | $3 \times 4$ |
| $A$ | attention weights | $4 \times 4$ |

The query followed through the video is **sat** (row 3).

## Part 0: Title (`IntroScene`)

Shows the scaled dot-product attention equation (Eq. 1 in the paper):

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

A caption states the toy sizes and that the weights are random and untrained.

## Part 1: Tokens become vectors (`EmbeddingScene`)

$$X = E + PE$$

- The four token chips appear, then each word slides into its row of $E$.
- $E$ is drawn from $\mathcal{N}(0, 1)$. A trained model would learn it.
- $PE$ is the sinusoidal encoding from Section 3.5:
  $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$ and
  $PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$.
- The caption says why it is needed: attention on its own is
  permutation-equivariant, so it has no notion of order.

## Part 2: Q, K and V (`QKVScene`)

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

- Shape note: $(4 \times 4)(4 \times 3) \rightarrow (4 \times 3)$.
- The three result matrices appear in blue, orange and green. Those colours
  stay attached to Q, K and V for the rest of the video.
- The "what this position looks for / offers / passes on" reading is shown
  with a caption that labels it as intuition. The maths only says these are
  three linear maps of the same $X$.

## Part 3: Scores (`ScoresScene`)

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

- Row 3 of $Q$ (the query for "sat") is boxed.
- Worked example: $q_{\text{sat}} \cdot k_{\text{cat}} = 0.44$, and
  $0.44 / \sqrt{3} = 0.25$. That value flies into cell (sat, cat) of the score
  heatmap, then the other 15 cells fill in.
- The heatmap uses a diverging scale: blue for negative scores, yellow for
  positive.
- The caption gives the paper's reason for $\sqrt{d_k}$: without it, dot
  products grow with $d_k$ and push softmax into regions with very small
  gradients (Section 3.2.1).

## Part 4: Softmax (`SoftmaxScene`)

$$\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}$$

- Score heatmap on the left, weight heatmap on the right. Each row moves
  across one at a time, which shows that softmax normalises per query.
- Every row ends with "= 1.00".
- The "sat" row is boxed: its largest weight is 0.35, on "the". Part 7 comes
  back to this number.

## Part 5: Weighted sum of values (`WeightedValuesScene`)

$$z_{\text{sat}} = \sum_j \alpha_{\text{sat},j}\, v_j, \qquad Z = AV$$

- The four value vectors are drawn as columns. Each is labelled with its
  weight (0.35, 0.29, 0.22, 0.15) and faded in proportion to it.
- The four columns slide together and become $z_{\text{sat}}$, shown with
  its values $(1.28, 0.36, -0.77)$.
- The caption notes that $Z = AV$ does this for every row at once.

## Part 6: Stacking layers (`LayersScene`)

$$X_{\ell+1} = \mathrm{LayerNorm}\left(X_\ell + \mathrm{Attention}(X_\ell)\,W^O\right)$$

- Left: the token vectors of $X_0, X_1, X_2, X_3$ projected to 2D. One PCA
  basis is fitted on all four layers together, so dots move only when the
  vectors change.
- Right: the attention weight heatmap of the layer that produced each step.
- On screen: "feed-forward sublayer and multiple heads omitted in this toy".
- Closing captions: with random weights the motion has no linguistic meaning,
  and after layer 1 the vector at position "sat" is already a mixture of all
  four tokens.

## Part 7: What attention weights do not tell you (`LimitationsScene`)

First, a demonstration on the same numbers:

| key | $\alpha_{\text{sat},j}$ | $\lVert \alpha_{\text{sat},j} v_j \rVert$ |
|---|---|---|
| the | **0.35** | 0.47 |
| cat | 0.29 | **0.80** |
| sat | 0.22 | 0.36 |
| down | 0.15 | 0.26 |

By weight, "sat" attends most to "the". By the size of what each token adds
to the output, "cat" comes first, because $\lVert v_{\text{cat}} \rVert = 2.78$
is about twice $\lVert v_{\text{the}} \rVert = 1.34$. This is the norm-based
analysis of Kobayashi et al. (2020).

Then a list, each item with its source:

1. A weight ignores the size of the vector it scales (Kobayashi et al., 2020).
2. Very different weights can give the same prediction (Jain & Wallace, 2019),
   though whether that undermines them is disputed (Wiegreffe & Pinter, 2019).
3. Residual paths carry information around attention (Abnar & Zuidema, 2020).
4. Deep positions hold mixtures, not single words (Brunner et al., 2020).
5. One heatmap is one head in one layer.

Closing line: attention weights show where one step reads from. Finding out
why a model makes a decision needs tests, such as removing or changing inputs
and measuring the effect. The caption points to Bibal et al. (2022) for an
overview of the debate.

## Part 8: References (`ReferencesScene`)

The seven papers cited above, listed on screen.
