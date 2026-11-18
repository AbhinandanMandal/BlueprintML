# Neural Style Transfer

From-scratch PyTorch implementation of **A Neural Algorithm of Artistic Style** (Gatys et al., 2015, [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)).

## Architecture

- `nst/losses.py`:
  - `gram_matrix(features)`: Computes channel-wise Gram matrices capturing style correlations.
  - `ContentLoss`: MSE loss between feature activations of the generated image and content target.
  - `StyleLoss`: MSE loss between Gram matrices of the generated image and style target.
- `nst/model.py`:
  - `VGGFeatureExtractor`: Extracts intermediate content and style feature maps using VGG-19 (or a custom `nn.Sequential` backbone for offline testing) with ImageNet normalization.
- `nst/train.py`:
  - `run_style_transfer(...)`: Optimizes the input image directly using `LBFGS` (or `Adam`), clamping pixel values to `[0, 1]`.

## Usage

```python
from nst.train import run_style_transfer

output = run_style_transfer(
    content_img,
    style_img,
    num_steps=300,
    style_weight=1e6,
    content_weight=1.0,
    lr=1.0,
)
```

## Offline Smoke Test & Unit Tests

Run the offline CLI smoke test on synthetic tensors (no pretrained weights download required):

```bash
python -m nst.train --smoke-test
```

Run the offline unit test suite:

```bash
python -m unittest test_nst.py
```
