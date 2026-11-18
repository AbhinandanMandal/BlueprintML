import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import ContentLoss, StyleLoss, gram_matrix
from .model import VGGFeatureExtractor


def run_style_transfer(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    extractor: Optional[VGGFeatureExtractor] = None,
    num_steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    lr: float = 1.0,
    optimizer_type: str = "lbfgs",
) -> torch.Tensor:
    """Optimize an input image to match content and style representations."""
    if extractor is None:
        extractor = VGGFeatureExtractor().to(content_img.device)
    extractor.eval()

    with torch.no_grad():
        target_content_feats, _ = extractor(content_img)
        _, target_style_feats = extractor(style_img)
        target_style_grams = {
            name: gram_matrix(feat).detach()
            for name, feat in target_style_feats.items()
        }
        target_content_feats = {
            name: feat.detach() for name, feat in target_content_feats.items()
        }

    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss()

    input_img = content_img.clone().detach().requires_grad_(True)

    opt_name = optimizer_type.lower()
    if opt_name == "lbfgs":
        optimizer: optim.Optimizer = optim.LBFGS([input_img], lr=lr)
    elif opt_name == "adam":
        optimizer = optim.Adam([input_img], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    step = [0]
    while step[0] < num_steps:
        def closure() -> torch.Tensor:
            with torch.no_grad():
                input_img.clamp_(0.0, 1.0)

            optimizer.zero_grad()
            c_feats, s_feats = extractor(input_img)

            c_loss = torch.tensor(0.0, device=input_img.device)
            for name, feat in c_feats.items():
                c_loss = c_loss + content_loss_fn(feat, target_content_feats[name])

            s_loss = torch.tensor(0.0, device=input_img.device)
            for name, feat in s_feats.items():
                s_loss = s_loss + style_loss_fn(feat, target_style_grams[name])

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()
            step[0] += 1
            return total_loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0.0, 1.0)

    return input_img.detach()


def _compute_total_loss(
    img: torch.Tensor,
    target_c: dict,
    target_s_grams: dict,
    extractor: VGGFeatureExtractor,
    content_weight: float,
    style_weight: float,
) -> float:
    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss()
    with torch.no_grad():
        c_feats, s_feats = extractor(img)
        c_loss = sum(
            content_loss_fn(feat, target_c[name]) for name, feat in c_feats.items()
        )
        s_loss = sum(
            style_loss_fn(feat, target_s_grams[name]) for name, feat in s_feats.items()
        )
        return float((content_weight * c_loss + style_weight * s_loss).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run offline smoke test on synthetic tensors",
    )
    parser.add_argument("--steps", type=int, default=20, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "adam"],
        help="Optimizer type",
    )
    args = parser.parse_args()

    if args.smoke_test:
        torch.manual_seed(0)
        backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        extractor = VGGFeatureExtractor(
            content_layers=["conv_2"],
            style_layers=["conv_1", "conv_2"],
            features=backbone,
            normalize=False,
        )
        content_img = torch.rand(1, 3, 16, 16)
        style_img = torch.rand(1, 3, 16, 16)

        with torch.no_grad():
            tc, _ = extractor(content_img)
            _, ts = extractor(style_img)
            ts_grams = {k: gram_matrix(v) for k, v in ts.items()}

        init_loss = _compute_total_loss(
            content_img, tc, ts_grams, extractor, 1.0, 1e3
        )
        output = run_style_transfer(
            content_img,
            style_img,
            extractor=extractor,
            num_steps=args.steps,
            style_weight=1e3,
            content_weight=1.0,
            lr=args.lr,
            optimizer_type=args.optimizer,
        )
        final_loss = _compute_total_loss(
            output, tc, ts_grams, extractor, 1.0, 1e3
        )
        print(f"Initial loss: {init_loss:.6f}")
        print(f"Final loss:   {final_loss:.6f}")
        print(f"Output shape: {tuple(output.shape)}, min={output.min():.4f}, max={output.max():.4f}")


if __name__ == "__main__":
    main()
