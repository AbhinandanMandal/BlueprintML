# MoCo v1 — Technical Specification (Stage A)

Paper: He et al., *Momentum Contrast for Unsupervised Visual Representation Learning*, CVPR 2020 (arXiv:1911.05722v3).
Verified against: the paper PDF and the official `facebookresearch/moco` source (`moco/builder.py`, `main_moco.py`,
read from the repo's `colab-notebook` branch; the `main` branch snapshot served by the sandbox proxy had an empty tree).
The official code was used only to verify details; this implementation is written independently.

## 1. Research problem

- **Positives and negatives.** Contrastive learning treats representation learning as instance discrimination:
  a query must be close to its positive key (another view of the same image) and far from keys of other images.
  Without negatives the trivial solution is a constant embedding.
- **Large, consistent dictionary.** More negatives give a better approximation of the softmax over the whole
  dataset. Keys should also come from encoders that change slowly, so keys in the dictionary stay comparable.
- **Batch-size limit.** If negatives come only from the current minibatch, the dictionary size equals the batch
  size, which is bounded by GPU memory. Taking the gradient through all keys (end-to-end) couples size to memory.
- **Queue.** Keys from earlier minibatches are kept in a FIFO queue. The dictionary size K is then decoupled
  from the batch size N. Each step enqueues N new keys and dequeues the N oldest.
- **Momentum encoder.** Queued keys cannot receive gradients, so the key encoder cannot be updated by
  backprop. Copying the query encoder each step makes old and new keys inconsistent (the paper reports this
  fails to converge). A slow EMA, `theta_k <- m*theta_k + (1-m)*theta_q`, keeps them consistent.

## 2. Architecture

```
 image x --T--> x_q ---> f_q (ResNet + fc) --L2norm--> q  [N,C]   (gradients flow)
        \--T--> x_k ---> f_k (ResNet + fc) --L2norm--> k+ [N,C]   (no_grad, momentum-updated)

 queue  [K,C]  (FIFO, L2-normalised keys from previous batches) = negatives k_1^- ... k_K^-

 l_pos = q . k+            [N,1]
 l_neg = q @ queue^T       [N,K]
 logits = cat([l_pos, l_neg]) / tau   [N,1+K]   positive at column 0
 loss   = CrossEntropy(logits, target=0)
 then:  SGD step on f_q  ->  theta_k <- m*theta_k + (1-m)*theta_q  ->  enqueue(k+)
```

Roles: `f_q` query encoder (trained by SGD); `f_k` key encoder (EMA of `f_q`); `q` query; `k+` positive key;
`k_i^-` negatives from the queue; `m` momentum (0.999); `tau` temperature (0.07).

## 3. Mathematical formulation

`L_q = -log[ exp(q.k+/tau) / ( exp(q.k+/tau) + sum_{i=1..K} exp(q.k_i^-/tau) ) ]`

This is exactly softmax cross-entropy over a (1+K)-way classification whose correct class is the positive.
Because `torch.nn.functional.cross_entropy` takes the class index, the positive must be placed at the column
that the target index points to; here column 0, so targets are all zeros. Putting the positive elsewhere
(or the wrong target) silently trains the wrong objective.

## 4. Equation-to-code map

| Paper | Code |
|---|---|
| Eq. 1 InfoNCE | `losses.moco_logits`, `losses.info_nce_loss` |
| Eq. 2 momentum update | `MoCo.momentum_update` |
| Dictionary as queue | `MoCo.queue`, `MoCo.queue_ptr`, `MoCo.enqueue` |
| Stop-gradient on keys | `requires_grad=False` on `encoder_k`, `@no_grad` key encode, `.detach()` |
| L2-normalised 128-d output | `MoCoEncoder.forward` |
| Shuffling BN | emulated on one device with `bn_groups > 1` (see below) |

## 5. Fidelity checklist

| Item | Paper | Official code | This implementation |
|---|---|---|---|
| Loss | InfoNCE, Eq. 1 | CE on cat(l_pos, l_neg)/T, labels 0 | same |
| m | 0.999 | `--moco-m 0.999` | 0.999 (paper) |
| tau | 0.07 | `--moco-t 0.07` | 0.07 |
| Output dim | 128 | `--moco-dim 128` | 128 |
| Queue K | 65536 (Table 1), 4096 (ablations) | default 65536 | 256 (smoke), 4096 (small run): **Colab choice** |
| Optimizer | SGD, momentum 0.9, wd 1e-4, lr 0.03 (batch 256) | same | same defaults, **adapted** for batch 32/64 |
| LR schedule | x0.1 at epochs 120, 160 of 200 | same (`--schedule 120 160`) | step at 60%/80% of run (**adapted**) |
| Aug (v1) | crop, color jitter, flip, grayscale | `RandomResizedCrop(224, scale=(0.2,1))`, `RandomGrayscale(0.2)`, `ColorJitter(0.4,0.4,0.4,0.4)`, flip | same ops; crop size 32 (**CIFAR adaptation**) |
| Backbone | ResNet-50, fc -> 128 | same, `fc` is the output layer | ResNet-18, fc -> 128, random init; 3x3 stem, no maxpool (**CIFAR adaptation**) |
| Shuffling BN | multi-GPU shuffle | `_batch_shuffle_ddp` | single-device emulation via `bn_groups` (**engineering choice**) |
| Update order | Alg. 1: SGD, then momentum update, then enqueue | momentum update at the *start* of `forward` (before key encode), enqueue at the end of forward, before the optimizer step | momentum update at start of step, as official |
| Queue insertion | enqueue N, dequeue N | `assert K % N == 0`, no wraparound | wraparound supported |
| Queue init | not specified | `randn`, normalised | same; the first K/N steps contain random negatives |

**Order note.** Paper pseudocode places the momentum update after the SGD step; the official code runs it at the
start of the next forward, which is equivalent up to the very first step (key encoder starts identical to the
query encoder). The task spec lists the official order, which this implementation follows.

**Uncertain / not verified.** The paper's exact ImageNet jitter strength is described in prose only; I use the
official v1 values above. Exact BN running-stat handling for `f_k`: official code momentum-updates only
`parameters()`, so key BN buffers come from their own forward passes; this implementation does the same.

## 6. Scope limits

CIFAR-10 at 32x32 with ResNet-18 and a few thousand images is a debugging and learning setup. Numbers from it are
not comparable to ImageNet results in the paper (60.6% linear top-1 with ResNet-50).
