import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None

try:
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
except Exception:
    efficientnet_b3 = None
    EfficientNet_B3_Weights = None


class SoftAttentionUnit(nn.Module):
    """Paper Eq.(1): attended = gamma * t * alpha + t."""

    def __init__(self, in_channels: int, K: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.K = K
        self.conv3d = nn.Conv3d(1, K, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        nn.init.kaiming_normal_(self.conv3d.weight, mode="fan_out", nonlinearity="relu")
        if self.conv3d.bias is not None:
            nn.init.zeros_(self.conv3d.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, C, H, W)
        b, c, h, w = t.shape
        x = t.unsqueeze(1)  # (B, 1, C, H, W)
        attn_logits = self.conv3d(x)  # (B, K, C, H, W)

        # Soft attention over kernels then aggregate to channel-wise alpha.
        alpha_k = torch.softmax(attn_logits, dim=1)
        alpha = alpha_k.mean(dim=1)  # (B, C, H, W)

        attended = self.gamma * t * alpha + t
        return attended


class SoftAttentionBlock(nn.Module):
    """Dual-path soft attention block from paper Fig.4a."""

    def __init__(self, in_channels: int, K: int = 16, dropout: float = 0.5):
        super().__init__()
        self.pool_a = nn.MaxPool2d(2, 2)
        self.attn = SoftAttentionUnit(in_channels=in_channels, K=K)
        self.pool_b = nn.MaxPool2d(2, 2)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path_a = self.pool_a(x)
        path_b = self.pool_b(self.attn(x))
        out = torch.cat([path_a, path_b], dim=1)
        out = self.drop(self.act(out))
        return out


class _TimmEffNetV2B3Trunk(nn.Module):
    """EfficientNetV2-B3 trunk with last 16 MBConv blocks excluded."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is not available")

        model = None
        for model_name in ["efficientnetv2_b3", "tf_efficientnetv2_b3", "efficientnet_b3"]:
            try:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
                break
            except Exception:
                model = None
        if model is None:
            raise RuntimeError("No compatible EfficientNet-B3 variant found in timm")

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        blocks = list(model.blocks)
        keep_n = max(1, len(blocks) - 16)
        self.blocks = nn.Sequential(*blocks[:keep_n])
        self.out_channels = blocks[keep_n - 1].conv_pwl.out_channels if hasattr(blocks[keep_n - 1], "conv_pwl") else 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x


class _TorchvisionEffNetB3Trunk(nn.Module):
    """Fallback trunk: torchvision efficientnet_b3 features[:-1]."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if efficientnet_b3 is None:
            raise ImportError("torchvision efficientnet_b3 is not available")
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained and EfficientNet_B3_Weights is not None else None
        model = efficientnet_b3(weights=weights)
        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.out_channels = 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SkinEfficientNetB3(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = True, K: int = 16, dropout: float = 0.5):
        super().__init__()

        if timm is not None:
            try:
                self.backbone = _TimmEffNetV2B3Trunk(pretrained=pretrained)
                feat_dim = self.backbone.out_channels
            except Exception:
                self.backbone = _TorchvisionEffNetB3Trunk(pretrained=pretrained)
                feat_dim = self.backbone.out_channels
        else:
            self.backbone = _TorchvisionEffNetB3Trunk(pretrained=pretrained)
            feat_dim = self.backbone.out_channels

        self.feat_dim = feat_dim
        self.sa = SoftAttentionBlock(feat_dim, K=K, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(feat_dim * 2),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.sa(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_groups(self, backbone_lr: float = 2e-5, head_lr: float = 1e-4) -> List[dict]:
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.sa.parameters(), "lr": head_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkinEfficientNetB3(num_classes=9, pretrained=False, K=16, dropout=0.5).to(device)
    x = torch.randn(2, 3, 300, 300, device=device)
    y = model(x)

    assert y.shape == (2, 9), f"Unexpected output shape: {tuple(y.shape)}"
    gamma = float(model.sa.attn.gamma.detach().cpu().item())
    print(f"output shape {tuple(y.shape)} ✓ | gamma={gamma:.4f} | PASSED")
