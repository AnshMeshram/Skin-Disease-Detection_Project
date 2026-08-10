from typing import List

import torch
import torch.nn as nn

from torchvision.models import inception_v3, Inception_V3_Weights


class SkinInceptionV3(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        # aux_logits must be True if using pretrained weights (torchvision quirk)
        model = inception_v3(weights=weights, aux_logits=True if pretrained else False)

        # Remove auxiliary classifier for simplicity and speed
        model.aux_logits = False
        model.AuxLogits = None

        feat_dim = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if isinstance(f, tuple):
            f = f[0]
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
