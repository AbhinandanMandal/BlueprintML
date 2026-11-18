import unittest

import torch
import torch.nn as nn

from nst.losses import ContentLoss, StyleLoss, gram_matrix
from nst.model import VGGFeatureExtractor
from nst.train import run_style_transfer


def _build_offline_extractor() -> VGGFeatureExtractor:
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    return VGGFeatureExtractor(
        content_layers=["conv_2"],
        style_layers=["conv_1", "conv_2"],
        features=backbone,
        normalize=False,
    )


class TestNeuralStyleTransfer(unittest.TestCase):
    def test_gram_matrix_shape_and_symmetry(self) -> None:
        features = torch.randn(1, 6, 8, 8)
        gram = gram_matrix(features)
        self.assertEqual(gram.shape, (6, 6))
        self.assertTrue(torch.allclose(gram, gram.t(), atol=1e-6))

    def test_content_loss_backward_gradients(self) -> None:
        target = torch.randn(1, 4, 8, 8)
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss_fn = ContentLoss()
        loss = loss_fn(x, target)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_style_loss_backward_gradients(self) -> None:
        target_feat = torch.randn(1, 4, 8, 8)
        target_gram = gram_matrix(target_feat)
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss_fn = StyleLoss()
        loss = loss_fn(x, target_gram)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_run_style_transfer_reduces_total_loss_offline(self) -> None:
        torch.manual_seed(42)
        extractor = _build_offline_extractor()
        content_img = torch.rand(1, 3, 16, 16)
        style_img = torch.rand(1, 3, 16, 16)

        content_weight = 1.0
        style_weight = 1e3

        with torch.no_grad():
            tc, _ = extractor(content_img)
            _, ts = extractor(style_img)
            ts_grams = {k: gram_matrix(v) for k, v in ts.items()}

        def total_loss(img: torch.Tensor) -> float:
            c_fn = ContentLoss()
            s_fn = StyleLoss()
            with torch.no_grad():
                cf, sf = extractor(img)
                cl = sum(c_fn(v, tc[k]) for k, v in cf.items())
                sl = sum(s_fn(v, ts_grams[k]) for k, v in sf.items())
                return float((content_weight * cl + style_weight * sl).item())

        initial_loss = total_loss(content_img)
        output = run_style_transfer(
            content_img,
            style_img,
            extractor=extractor,
            num_steps=15,
            style_weight=style_weight,
            content_weight=content_weight,
            lr=0.5,
            optimizer_type="lbfgs",
        )
        final_loss = total_loss(output)

        self.assertEqual(output.shape, content_img.shape)
        self.assertGreaterEqual(output.min().item(), 0.0)
        self.assertLessEqual(output.max().item(), 1.0)
        self.assertLess(final_loss, initial_loss)


if __name__ == "__main__":
    unittest.main()
