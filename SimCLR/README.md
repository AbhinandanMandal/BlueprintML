# SimCLR reproduction (scaled-down, PyTorch)

A from-scratch, modular PyTorch implementation of **SimCLR** — *A Simple Framework for Contrastive Learning of
Visual Representations* (Chen, Kornblith, Norouzi, Hinton, ICML 2020) — designed to run on a Google Colab GPU
using Python files only.

> **Scope.** The experiments here are *scaled-down* (CIFAR-10, ResNet-18, small batches, short schedules). They
> validate the pipeline and allow lightweight representation-learning experiments. They are **not** a
> reproduction of the paper's ImageNet results, and CIFAR-10 numbers are not comparable to ImageNet numbers.
> This repository ships **no benchmark numbers**; run the commands below to produce your own.

- Paper: https://arxiv.org/abs/2002.05709 

## Methodology (what is implemented)

1. **Augmentation** – two independent draws of one stochastic transform per image: random resized crop, flip,
   colour jitter (p = 0.8, strength `s`), random grayscale (p = 0.2), optional Gaussian blur.
2. **Encoder** `f` – randomly initialised ResNet-18 (default) or ResNet-50 with a CIFAR stem (3×3 stride-1 conv,
   no max-pool); returns the pooled representation `h` (512-d / 2048-d), no classifier.
3. **Projection head** `g` – `Linear → BatchNorm → ReLU → Linear` (2048 hidden, 128 out), `z = g(h)`.
4. **NT-Xent loss** on `z` (L2-normalised, temperature τ = 0.5), 2N−2 negatives per anchor, no memory bank.
5. **Optimisation** – SGD+momentum (default) or LARS, linear warm-up + cosine decay, per-iteration, learning rate
   scaled linearly with batch size.
6. **Linear evaluation** – projection head discarded, encoder frozen, single linear layer trained on `h`.

## Repository structure

```
configs/            default.yaml (smoke test), research.yaml (10k-image run)
src/simclr/
  config.py         dataclass configs, validation, CLI generation
  augmentations.py  two-view augmentation + visualisation
  datasets.py       CIFAR-10 / synthetic loading, deterministic subsets, label-free pretrain dataset
  models.py         encoder f, projection head g
  losses.py         NT-Xent
  optimizers.py     LARS, SGD param groups, warm-up+cosine scheduler
  engine.py         train_one_epoch, fit (resume, logging, checkpoints), kNN monitor
  checkpoint.py     atomic save/load, resume compatibility checks
  evaluation.py     feature extraction, kNN, linear probe, frozen-encoder check
  metrics.py        accuracy, macro-F1, per-class accuracy, confusion matrix
  utils.py          seeding, device, RNG state, CSV/JSON logging
scripts/            train.py, linear_eval.py, evaluate.py (kNN), visualize_augmentations.py
tests/              unit tests (pytest)
```

## Paper-to-code mapping

| Paper component | Code |
|---|---|
| Stochastic augmentation `t(x), t'(x)` | `augmentations.build_train_transform`, `TwoViewTransform` |
| Base encoder `f(·)` → `h` | `models.build_encoder`, `SimCLRModel.encode` |
| Projection head `g(·)` → `z` | `models.ProjectionHead` |
| NT-Xent, Eq. (1) | `losses.nt_xent_loss` |
| Training step (2N views, loss on `z`) | `engine.train_one_epoch` |
| LARS, linear LR scaling, warm-up, cosine | `optimizers.LARS`, `TrainConfig.lr`, `WarmupCosineScheduler` |
| Linear evaluation protocol | `evaluation.run_linear_evaluation` |

## Setup on Google Colab (terminal)

Select **Runtime → Change runtime type → GPU**, open a terminal, and get the repository onto the VM (e.g.
`git clone <your-repo-url>` or upload/unzip it), then:

```bash
cd simclr-reproduction
python -m pip install -r requirements.txt    # Colab's preinstalled torch/torchvision are normally kept
python -m pytest -q
```

Colab's local disk is wiped when the session ends. To persist results, mount Google Drive in a Colab *notebook*
cell (`from google.colab import drive; drive.mount('/content/drive')`) and pass an `--output-dir` under
`/content/drive/MyDrive/...`. (This repo does not include a notebook.)

### Smoke test (Configuration A: 2,000 images, 3 epochs, batch 64)

```bash
python scripts/train.py \
    --dataset cifar10 --data-dir ./data --output-dir ./outputs/smoke_test \
    --train-subset-size 2000 --epochs 3 --batch-size 64 --encoder resnet18 \
    --projection-dim 128 --temperature 0.5 --seed 42
```

Equivalent: `python scripts/train.py --config configs/default.yaml`.

### Small research experiment (Configuration B: 10,000 images)

```bash
python scripts/train.py \
    --dataset cifar10 --data-dir ./data --output-dir ./outputs/research_run \
    --train-subset-size 10000 --epochs 30 --batch-size 128 --encoder resnet18 \
    --projection-dim 128 --temperature 0.5 --seed 42 --eval-every 5 --eval-subset-size 0
```

Use `--train-subset-size 0` for all 50,000 training images. Do **not** resume the 3-epoch smoke run as if it were
the 30-epoch run: start the research run in its own `--output-dir` (a checkpoint from a different configuration
is rejected).

### Checkpoints, resume, extending a run

`last_checkpoint.pt` is written every `--save-every` epochs and at the end; `best_checkpoint.pt` tracks the best
kNN accuracy (or lowest training loss when `--eval-every 0`). Checkpoints contain encoder, projector, optimizer,
scheduler, AMP scaler, epoch, global step, best metric, config, seed, Python/NumPy/PyTorch/CUDA RNG states and
software versions, and are written atomically.

```bash
# Colab disconnected? Re-run the same command with --resume auto (continues from <output-dir>/last_checkpoint.pt)
python scripts/train.py ...same arguments... --resume auto

# Train longer than originally planned (e.g. 30 -> 50 epochs)
python scripts/train.py ...same arguments... --epochs 50 --resume auto --extend-schedule
```

Fields that define the experiment (data, model, batch size, optimizer, temperature, …) must match the
checkpoint. Changing `--epochs` requires `--extend-schedule`; the cosine schedule is rebuilt for the new length,
so the learning rate may jump at the resume point.

### Linear evaluation

```bash
python scripts/linear_eval.py \
    --checkpoint ./outputs/smoke_test/last_checkpoint.pt \
    --data-dir ./data --output-dir ./outputs/linear_eval
```

Defaults: whole labelled training split (10% held out for monitoring), full official test split, Adam, 100
epochs. The script verifies (by checksum) that the encoder did not change. A quick kNN diagnostic:
`python scripts/evaluate.py --checkpoint ...`. Augmentation preview:
`python scripts/visualize_augmentations.py --output ./outputs/aug_pairs.png`.

## Testing

```bash
python -m pytest -q
```

Covers: loss vs. an independent loop-based reference (several N and τ), similarity-matrix shape, masked
diagonal, positive indices, gradients reaching head and encoder, model shapes, augmentations (independence,
determinism), deterministic subsets, split disjointness, LARS/scheduler behaviour, one full optimisation step,
fixed-batch overfitting, metrics, frozen-encoder linear evaluation, checkpoint save/restore/resume/extend, and
incompatible-resume rejection. CIFAR-10-on-disk and CUDA/AMP tests skip automatically when unavailable.

## Outputs

`<output-dir>/`: `config.json`, `environment.json` (library/GPU versions), `metrics.csv` (epoch, global step,
mean loss, lr used in the last step, epoch time, images/s where "images" = source images each yielding two
views, peak GPU MB, kNN accuracy), `summary.json`, `last_checkpoint.pt`, `best_checkpoint.pt`.
Linear evaluation: `linear_eval_results.json` (top-1, loss, macro-F1, per-class accuracy, confusion matrix,
feature dim, sample counts) and `confusion_matrix.csv`.

## Reproducibility

Python/NumPy/PyTorch/CUDA are seeded; subset selection uses its own seed (`--subset-seed`, default `--seed`);
the shuffle generator is re-seeded each epoch from `seed + epoch`. `--deterministic` requests deterministic
kernels. Bit-exact reproducibility is **not** guaranteed (non-deterministic CUDA ops, AMP, DataLoader workers,
different library versions).
Label separation: pre-training uses a dataset wrapper that drops labels, drawn from the **train** split; the
kNN monitor and linear probe use labels for evaluation only, and test data come from the official **test** split.

## Deviations from the paper and known limitations

- CIFAR-10 / ResNet-18 / 32×32 instead of ImageNet / ResNet-50 / 224×224; short schedules and small batches
  (batch 64–128 gives 126–254 negatives per anchor vs. 8,190 at batch 4096), which weakens the contrastive task.
- **SGD+momentum is the default optimiser**; LARS is implemented (unit-tested on toy problems only) and
  selectable, but it has not been run at scale here. Default learning rates/weight decay for SGD are untuned.
- Linear probe is trained on precomputed features of un-augmented images (paper's protocol trains on augmented
  images); features are optionally standardised (still linear).
- Projection head uses BatchNorm only after the hidden layer. Any additional normalisation in the official code
  is not replicated. The official repository was **not** consulted while writing this code.
- CIFAR-specific settings (colour strength 0.5, no blur, 3×3 stem, no max-pool, τ = 0.5) were implemented from
  the task specification and my recollection of the paper's appendix; the sandbox could only fetch the arXiv
  abstract, so **these were not verified against the paper text**. Please check them against Appendix B.
- Default crop scale (0.08–1.0) is the paper's ImageNet setting; CIFAR works often use a narrower range
  (configurable via `--crop-scale-min`).
- `--image-size > 32` upsamples CIFAR images, increasing compute roughly with the pixel count.
- Weight decay and LARS adaptation are skipped for biases/BatchNorm parameters in both optimisers.
- When AMP's GradScaler skips a step because of inf gradients, the LR scheduler still advances.
- Checkpoints are loaded with `weights_only=False` (they contain RNG state); only load files you created.
- The `synthetic` dataset exists only to test the plumbing offline; results on it (e.g. perfect kNN accuracy,
  since classes differ by colour) say nothing about representation quality.



