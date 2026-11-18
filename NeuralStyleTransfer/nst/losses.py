from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.size()
    reshaped = features.view(b * c, h * w)
    gram = torch.mm(reshaped, reshaped.t())
    return gram.div(b * c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.target = target.detach() if target is not None else None
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self, input: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if target is not None:
            return F.mse_loss(input, target.detach())
        if self.target is None:
            raise ValueError("target must be provided either at init or forward.")
        self.loss = F.mse_loss(input, self.target.detach())
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.target_gram = (
            gram_matrix(target_feature).detach()
            if target_feature is not None
            else None
        )
        self.loss: torch.Tensor = torch.tensor(0.0)

    def forward(
        self, input: torch.Tensor, target_gram: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if target_gram is not None:
            return F.mse_loss(gram_matrix(input), target_gram.detach())
        if self.target_gram is None:
            raise ValueError("target_gram must be provided either at init or forward.")
        self.loss = F.mse_loss(gram_matrix(input), self.target_gram.detach())
        return input
