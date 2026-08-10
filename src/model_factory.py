import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from src.convnext_tiny import SkinConvNeXtTiny
from src.efficientnet_b3 import SkinEfficientNetB3
from src.inception_v3 import SkinInceptionV3


def build_model(name: str, num_classes: int = 9, pretrained: bool = True):
    name = name.lower()
    if name == "efficientnet_b3":
        return SkinEfficientNetB3(num_classes=num_classes, pretrained=pretrained)
    if name == "inception_v3":
        return SkinInceptionV3(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_tiny":
        return SkinConvNeXtTiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}")


def build_optimizer(model, base_lr: float, head_lr: float, weight_decay: float):
    if hasattr(model, "get_param_groups"):
        params = model.get_param_groups(backbone_lr=base_lr, head_lr=head_lr)
    else:
        params = [{"params": model.parameters(), "lr": base_lr}]
    return AdamW(params, weight_decay=weight_decay)


def build_scheduler(
    optimizer,
    total_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
    cosine_restart: bool = True,
):
    if cosine_restart:
        t0 = max(5, total_epochs // 4)
        return CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=min_lr)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
