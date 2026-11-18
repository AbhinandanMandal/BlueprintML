from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalization(nn.Module):
    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class VGGFeatureExtractor(nn.Module):
    DEFAULT_CONTENT_LAYERS = ["conv_4"]
    DEFAULT_STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    def __init__(
        self,
        content_layers: Optional[List[str]] = None,
        style_layers: Optional[List[str]] = None,
        features: Optional[nn.Sequential] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.content_layers = (
            list(content_layers)
            if content_layers is not None
            else list(self.DEFAULT_CONTENT_LAYERS)
        )
        self.style_layers = (
            list(style_layers)
            if style_layers is not None
            else list(self.DEFAULT_STYLE_LAYERS)
        )
        self.normalize = normalize
        self.norm = Normalization()

        if features is None:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        else:
            vgg = features

        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        target_layers = set(self.content_layers) | set(self.style_layers)
        named_layers: List[Tuple[str, nn.Module]] = []
        conv_idx = 0
        relu_idx = 0
        pool_idx = 0
        bn_idx = 0
        last_needed = -1

        for idx, layer in enumerate(vgg.children()):
            if isinstance(layer, nn.Conv2d):
                conv_idx += 1
                name = f"conv_{conv_idx}"
            elif isinstance(layer, nn.ReLU):
                relu_idx += 1
                name = f"relu_{relu_idx}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                pool_idx += 1
                name = f"pool_{pool_idx}"
            elif isinstance(layer, nn.BatchNorm2d):
                bn_idx += 1
                name = f"bn_{bn_idx}"
            else:
                name = f"layer_{idx}"

            named_layers.append((name, layer))
            if name in target_layers:
                last_needed = len(named_layers) - 1

        if last_needed >= 0:
            named_layers = named_layers[: last_needed + 1]

        self.layers = nn.ModuleDict(named_layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.normalize:
            x = self.norm(x)

        content_features: Dict[str, torch.Tensor] = {}
        style_features: Dict[str, torch.Tensor] = {}
        content_set = set(self.content_layers)
        style_set = set(self.style_layers)

        for name, layer in self.layers.items():
            x = layer(x)
            if name in content_set:
                content_features[name] = x
            if name in style_set:
                style_features[name] = x

        return content_features, style_features
