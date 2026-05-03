from typing import List

import torch
import torch.nn as nn

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class SkinConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)

        feat_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()

        self.backbone = model
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return self.head(f)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, backbone_lr: float = 2e-5, head_lr: float = 1e-4) -> List[dict]:
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
