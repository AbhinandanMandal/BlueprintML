# MoCo v1 from scratch (PyTorch, CIFAR-10, Colab-sized)

Independent implementation of *Momentum Contrast for Unsupervised Visual Representation Learning* (He et al., CVPR 2020).
No pretrained weights and no third-party MoCo code. See [SPEC.md](https://github.com/AbhinandanMandal/BlueprintML/blob/master/MoCo/SPEC.md) for the paper summary, equation map and fidelity table. Check **[MoCo Technical Report](https://github.com/AbhinandanMandal/ML-Whitepapers)**.
<img width="950" height="501" alt="download" src="https://github.com/user-attachments/assets/596f7378-02c9-453f-a0aa-98a8d85762c9" />

## Layout
`config.py` single validated config · `datasets.py` two-view CIFAR-10 + fixed subsets · `models/encoder.py` ResNet + fc→128 + L2 norm ·
`models/moco.py` queue, momentum update, logits · `losses.py` InfoNCE · `train.py` step/loop/checkpoint/resume ·
`evaluate.py` kNN + linear probe · `tests/` unit tests · `moco_v1_colab.ipynb` Colab notebook · `configs/*.yaml`.

## Run
```bash
pip install -r requirements.txt
python -m pytest moco_reproduction/tests -q                     # must pass first (run from the parent directory)
python -m moco_reproduction.run_experiment --mode smoke         # Mode A: 1,000 images, K=256, 20 steps
python -m moco_reproduction.run_experiment --mode small --linear-epochs 100   # Mode B: 5,000 images, K=4096, 10 epochs
```
In Colab, upload the notebook and run it top to bottom on a GPU runtime. Outputs: `*_metrics.csv`, `*_loss.png`, `*_lr.png`,
`*_results.json`, and `checkpoints/*_latest.pt`.

## Status of verification (read this before trusting any number)
- **Executed here (CPU sandbox, PyTorch 2.14):** 19 unit tests pass; a 20-step smoke run on a *synthetic* image set completed; checkpoint reload matched exactly; a 2→4 epoch resume continued the step counter and LR schedule.
- **Not executed here:** anything on real CIFAR-10 (the sandbox cannot reach the download server) and anything on a GPU. No CIFAR-10 accuracy exists yet. Run the notebook to produce it.
- Synthetic data is for mechanics only; its kNN/linear numbers are meaningless and must not be reported.
- Passing tests verify implementation correctness; they say nothing about empirical reproduction fidelity.

## Adaptations (not from the paper)
ResNet-18 with 3x3 stem and no max-pool; 32x32 crops; CIFAR mean/std; K and lr schedule scaled for a short run; optional `bn_groups>1`
emulates Shuffling BN on one device; queue wraparound supported. Known behaviour: the first K/N steps contain random negatives (queue
initialised as random unit vectors, as in the official code), and early loss can be near zero then jump to ≈ log(1+K) once real keys fill the queue.

## Troubleshooting
CUDA OOM → lower `batch_size` (changes the setup). `device='cuda'` without a GPU raises with instructions. Non-finite loss raises
`FloatingPointError`. Checkpoint with a different `queue_size` raises a clear `ValueError`. Interrupted runs save a checkpoint; resume with `resume_from=`.

## Suggested ablations (one variable at a time, baseline immutable)
Momentum m ∈ {0.9, 0.99, 0.999}; queue K ∈ {256, 1024, 4096}; temperature τ ∈ {0.07, 0.2}; batch 32 vs 64; seeds {42, 43, 44}; remove color jitter.
For each: record loss curve, kNN top-1, linear-probe accuracy against the random-init baseline; each run is a few minutes on a Colab GPU; differences within seed noise mean nothing.
