from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)

        ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
        ce = torch.clamp(ce, max=100.0)

        p_t = torch.exp(-ce)
        loss = torch.pow(1.0 - p_t, self.gamma) * ce
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        out = loss.mean()
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, smoothing: float = 0.1):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)

        log_p = F.log_softmax(logits, dim=1)
        log_p = torch.clamp(log_p, min=-100.0)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        if alpha is not None:
            true_dist = true_dist * alpha.unsqueeze(0)
            true_dist = true_dist / (true_dist.sum(dim=1, keepdim=True) + 1e-8)

        # Per-sample smoothed CE (equivalent to KL up to additive constant).
        ce = -(true_dist * log_p).sum(dim=1)
        ce = torch.clamp(ce, max=100.0)

        p = torch.softmax(logits, dim=1)
        p_t = (p * true_dist).sum(dim=1)
        p_t = torch.clamp(p_t, 1e-7, 1.0 - 1e-7)

        loss = torch.pow(1.0 - p_t, self.gamma) * ce
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        out = loss.mean()
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


class WeightedCE(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)
        loss = F.cross_entropy(logits, targets, weight=alpha, reduction="mean")
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        return loss


def _prepare_alpha(config: dict, class_weights, device: str):
    use_alpha = bool(config.get("use_alpha", False))
    if not use_alpha or class_weights is None:
        return None

    alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)
    alpha = torch.clamp(alpha, 0.01, 100.0)
    alpha = alpha / (alpha.mean() + 1e-8)
    alpha = torch.where(torch.isfinite(alpha), alpha, torch.ones_like(alpha))
    return alpha


def build_loss_function(config: dict, class_weights=None, device: str = "cuda"):
    alpha = _prepare_alpha(config, class_weights, device)

    loss_name = str(config.get("loss", "focal_smooth")).lower()
    gamma = float(config.get("focal_gamma", 2.0))
    smoothing = float(config.get("label_smoothing", 0.1))

    if loss_name == "focal_smooth":
        return LabelSmoothingFocalLoss(gamma=gamma, alpha=alpha, smoothing=smoothing)
    if loss_name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha)
    if loss_name == "weighted_ce":
        return WeightedCE(alpha=alpha)

    raise ValueError(f"Unknown loss type: {loss_name}")


def verify_loss(config: dict, class_weights, device: str = "cpu") -> bool:
    loss_fn = build_loss_function(config, class_weights, device=device)
    logits = torch.randn(8, 9, device=device)
    labels = torch.randint(0, 9, (8,), device=device)
    val = loss_fn(logits, labels)
    assert torch.isfinite(val), f"LOSS IS NaN/Inf: {val}"
    assert val.item() > 0, f"LOSS IS ZERO: {val}"
    print(f"  Loss sanity check PASSED: {val.item():.4f}")
    return True


if __name__ == "__main__":
    cfg = {
        "loss": "focal_smooth",
        "focal_gamma": 2.0,
        "label_smoothing": 0.1,
        "use_alpha": True,
    }
    weights = [1.0, 0.2, 0.8, 2.0, 1.1, 4.5, 4.0, 2.8, 1.5]
    verify_loss(cfg, weights, device="cpu")
