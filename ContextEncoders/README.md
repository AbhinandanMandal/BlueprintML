# Context Encoders (CelebA 128×128, PyTorch)

A PyTorch implementation of **Context Encoders: Feature Learning by Inpainting**, refactored from a
research notebook into an importable package with training and inference scripts.
A masked image goes in; the network predicts the missing pixels, trained with a masked L2
reconstruction loss plus an adversarial loss.

> *Context Encoders: Feature Learning by Inpainting.* CVPR 2016. [arXiv:1604.07379](https://arxiv.org/abs/1604.07379)


This is a **small-scale implementation for learning purposes** (8k training images, 20 epochs,
128×128 inputs). It is not a reproduction of the paper's ImageNet/Paris StreetView experiments and makes no claim of matching them.

## Architecture
| Part | Definition (`src/context_encoders/model.py`) |
|---|---|
| Encoder | AlexNet-style: conv7×7/2 → pool → conv5×5 → pool → 3×conv3×3 → pool; each conv is Conv–BatchNorm–LeakyReLU(0.2). 3×128×128 → 256×8×8 |
| Bottleneck | `ChannelWiseFC` (one 64×64 FC per channel, `einsum`) followed by a 1×1 conv |
| Decoder | 4 × (ConvTranspose 4×4/2 – BN – ReLU), then ConvTranspose 3×3/1 + Tanh → 3×128×128 |
| Discriminator | 5 × Conv 4×4/2 (BN except first, LeakyReLU 0.2) → Linear → one logit |

Generator: 4,343,171 parameters (checked in the tests). The generator input is the masked image with hole pixels set to 0 (in the normalised [-1, 1] space); no mask channel is fed to the network.

**Losses** (`losses.py`): masked L2 per sample divided by (hole pixels × channels), batch-mean;
non-saturating BCE GAN loss; generator loss `0.999·L_rec + 0.001·L_adv`.
**Training schedule** (`engine.py`): epochs 1–2 reconstruction only, adversarial term from epoch 3; one random mask type
(central / block / region, each ≈ ≤25 % of the image) per batch; Adam, lr 2e-4, betas (0.5, 0.999).

## Repository structure
```
configs/default.py            all default hyper-parameters (traceable to the notebook)
src/context_encoders/
    model.py                  Encoder, ChannelWiseFC, Decoder, ContextEncoder, Discriminator
    masking.py                mask_central / mask_block / mask_region, make_mask
    losses.py                 ReconstructionLoss, AdversarialLoss, JointLoss
    dataset.py                CelebADataset, transform, train/val split + loaders
    engine.py                 train_one_epoch, fit
    evaluate.py               masked-L2 + PSNR evaluation, sample grids, curves
    utils.py                  seeding, device, checkpoint save/load
scripts/train.py, scripts/infer.py
tests/                        model, masking, loss, CPU smoke test, notebook-equivalence test
tests/results                 trained model outputs
```

## Setup
```bash
pip install -r requirements.txt        # torch, torchvision, numpy, Pillow, matplotlib (+ pytest)
pip install -e .                       # optional; scripts also work without installing
```
Colab: `!pip install -q -r requirements.txt` then run the commands below with `!python ...`.

## Data
Download CelebA ("Align&Cropped" images) and point `--data-dir` at the folder holding the `.jpg` files
(`img_align_celeba`). Labels are not used. Images are resized to 128×128 and normalised to [-1, 1].
By default 8,000 random images are used for training and 1,000 for validation (the notebook also uses this
validation set as its "test" set). The split indices are saved to `<output-dir>/split_indices.json`.

## Training
```bash
python scripts/train.py --data-dir /path/to/img_align_celeba --output-dir outputs/run1
# quick smoke test
python scripts/train.py --data-dir /path/to/img_align_celeba --output-dir outputs/smoke \
    --epochs 2 --max-steps 5 --mask-kinds central --train-size 256 --val-size 64
```
Outputs in `--output-dir`: `context_encoder_last.pt` (overwritten each epoch), `history.json`, `curves.png`,
`samples/*.png`, `config.json`, `split_indices.json`. The script refuses to overwrite an existing checkpoint
unless `--overwrite` is passed. Run `python scripts/train.py --help` for all options.

## Inference
```bash
python scripts/infer.py --checkpoint outputs/run1/context_encoder_last.pt \
    --images a.jpg b.jpg --mask central --output-dir results
```
Writes `results/inpainting_<mask>.png` with rows: masked input / raw prediction / composite / ground truth.
Choose `--mask central|block|region`. The model was trained on aligned 128×128 faces; unaligned images will look worse.

## Checkpoints
Format: `{"G": state_dict, "D": state_dict, "epoch": int}` (identical to the notebook; optimizer states are *not* saved,
so training cannot be resumed exactly). Loading uses `torch.load(..., weights_only=True)`.
```python
from context_encoders.model import ContextEncoder
from context_encoders.utils import load_checkpoint
G = ContextEncoder(); epoch = load_checkpoint("context_encoder_last.pt", G)
```
**Compatibility with the original trained weights:** parameter names and shapes of `ContextEncoder` and `Discriminator` match the
notebook classes exactly (strict `load_state_dict` between the two is tested). A checkpoint file produced by the notebook run was
**not available** during refactoring, so loading a real notebook checkpoint has not been tested.

## Key hyper-parameters (defaults = notebook)
| Name | Value | | Name | Value |
|---|---|---|---|---|
| epochs | 20 | | batch size | 32 |
| adversarial start | epoch 3 (`--adv-start-epoch 2`) | | train / val images | 8000 / 1000 |
| λ_rec / λ_adv | 0.999 / 0.001 | | lr, betas | 2e-4, (0.5, 0.999) |
| mask kinds | central, block, region | | seeds | torch 0, mask RNG 0, eval 123 |

## Reproducibility notes
- The notebook's train/val split used an **unseeded** `np.random.permutation`, so it cannot be recovered. Here the split is
  controlled by `--split-seed` (default 0) and saved to disk; results will differ slightly from the notebook run.
- GPU training uses `cudnn.benchmark = True` (as in the notebook) and multi-worker loading, so runs are not bit-wise reproducible.
- For reference, the notebook's own log at epoch 20 (Kaggle T4) reads: validation masked-L2 0.0784, PSNR 27.80 dB
  (mean over the three mask types, 8 val batches, fixed mask seed). These numbers come from that notebook run and were not re-generated here.
  The validation set doubles as test set, so treat them as a sanity check rather than a benchmark.

## Differences from the notebook (engineering only)
`plt.show()` calls became saved PNGs; paths and options moved to CLI/config; the data split is seedable; `CelebADataset` raises
clear errors for missing folders / unreadable images; checkpoint writes are atomic; the training script refuses to overwrite checkpoints;
`pin_memory` is enabled only on CUDA. Model, losses, masks, optimiser and training schedule are unchanged.

## Tests
```bash
pytest                                                                  # unit + CPU smoke tests
CONTEXT_ENCODERS_NOTEBOOK=/path/to/context-encoders.ipynb pytest tests/test_equivalence.py   # compare with the original notebook
```
The equivalence test executes the notebook's own model/mask/loss/training cells on synthetic CPU data and compares weights, masks,
losses, training history and final generator weights.

## Status and limitations
Small-scale run (8k images, 20 epochs); only central, block and random-region masks on 128×128 CelebA faces; no feature-learning /
pre-training evaluation from the paper; PSNR and masked-L2 are the only metrics.

## Citation
```bibtex
@inproceedings{pathak2016context,
  title={Context encoders: Feature learning by inpainting},
  author={Pathak, Deepak and Krahenbuhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2536--2544},
  year={2016}
}
```
