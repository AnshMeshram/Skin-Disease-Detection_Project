from pathlib import Path
from typing import Any
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation_classical import segment_grabcut, segment_otsu


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x2 with optional residual connection."""

    def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
        super().__init__()
        self.residual = residual and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.residual:
            y = y + x
        return y


class Down(nn.Module):
    """MaxPool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample with bilinear or transposed conv, concat skip, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net for binary lesion segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear=bilinear)
        self.up2 = Up(512, 128, bilinear=bilinear)
        self.up3 = Up(256, 64, bilinear=bilinear)
        self.up4 = Up(128, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        prob = self.forward(x)
        return (prob >= threshold).float()


class DiceLoss(nn.Module):
    """Dice loss for class-imbalanced segmentation."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred_prob.view(pred_prob.size(0), -1)
        tgt = target.view(target.size(0), -1)
        intersection = (pred * tgt).sum(dim=1)
        union = pred.sum(dim=1) + tgt.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """0.5*BCE + 0.5*Dice."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(pred_prob, target) + 0.5 * self.dice(pred_prob, target)


def _seg_metrics(pred_prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> tuple[float, float, float]:
    pred = (pred_prob >= thr).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    iou = tp / max(tp + fp + fn, 1e-6)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1e-6)
    pix_acc = (pred == target).float().mean().item()
    return iou, dice, pix_acc


class UNetTrainer:
    def __init__(self, model: nn.Module, config: dict, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = CombinedSegLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(config.get("unet_lr", 1e-4)),
            weight_decay=float(config.get("unet_weight_decay", 1e-5)),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        self.scaler = GradScaler(enabled=(device == "cuda" and bool(config.get("amp", True))))

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        n = 0

        for images, masks in tqdm(dataloader, desc="UNet train", leave=False):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                pred = self.model(images)
                loss = self.criterion(pred, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            iou, dice, _ = _seg_metrics(pred.detach(), masks)
            total_loss += float(loss.item())
            total_iou += iou
            total_dice += dice
            n += 1

        self.scheduler.step()
        return total_loss / max(n, 1), total_iou / max(n, 1), total_dice / max(n, 1)

    def val_epoch(self, dataloader: DataLoader) -> tuple[float, float, float, float, list[float]]:
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        per_image_iou: list[float] = []
        n = 0

        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="UNet val", leave=False):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                pred = self.model(images)
                loss = self.criterion(pred, masks)

                iou, dice, acc = _seg_metrics(pred, masks)
                per_image_iou.append(iou)
                total_loss += float(loss.item())
                total_iou += iou
                total_dice += dice
                total_acc += acc
                n += 1

        return (
            total_loss / max(n, 1),
            total_iou / max(n, 1),
            total_dice / max(n, 1),
            total_acc / max(n, 1),
            per_image_iou,
        )


def train_unet(config: dict, pseudo_mask_dir: str, out_dir: str = "outputs/unet") -> dict[str, Any]:
    """Train UNet from pseudo masks with 80/20 split."""
    from src.dataset import UNetDataset, build_dataframe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(config, prefer_preprocessed=True)
    if "mask_path" not in df.columns:
        df["mask_path"] = [str(Path(pseudo_mask_dir) / f"{img}.png") for img in df["image"]]

    valid = df[df["mask_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    if len(valid) < 10:
        raise RuntimeError("Not enough pseudo masks found for U-Net training")

    train_df = valid.sample(frac=0.8, random_state=42)
    val_df = valid.drop(train_df.index)

    train_ds = UNetDataset(train_df)
    val_ds = UNetDataset(val_df)
    train_dl = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=int(config.get("batch_size", 8)), shuffle=False, num_workers=0)

    model = UNet(n_channels=3, n_classes=1).to(device)
    trainer = UNetTrainer(model, config, device)

    best_iou = -1.0
    history = []
    epochs = int(config.get("unet_epochs", 30))

    for epoch in range(1, epochs + 1):
        tr_loss, tr_iou, tr_dice = trainer.train_epoch(train_dl)
        va_loss, va_iou, va_dice, va_acc, _ = trainer.val_epoch(val_dl)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_iou": tr_iou,
                "train_dice": tr_dice,
                "val_loss": va_loss,
                "val_iou": va_iou,
                "val_dice": va_dice,
                "val_acc": va_acc,
            }
        )

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(model.state_dict(), out_path / "best_unet.pth")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_path / "training_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    plt.plot(hist_df["epoch"], hist_df["val_iou"], label="val_iou")
    plt.plot(hist_df["epoch"], hist_df["val_dice"], label="val_dice")
    plt.legend()
    plt.xlabel("Epoch")
    plt.title("U-Net training curves")
    plt.tight_layout()
    plt.savefig(out_path / "training_curves.png", dpi=300)
    plt.close()

    return {"best_iou": best_iou, "checkpoint": str(out_path / "best_unet.pth")}


def predict_masks_batch(
    model: nn.Module,
    image_paths: list[str],
    device: str = "cuda",
    batch_size: int = 16,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Batch inference and save masks to outputs/unet_masks."""
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = PROJECT_ROOT / "outputs/unet_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[torch.Tensor] = []
    ok_paths: list[str] = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        tensors.append(t)
        ok_paths.append(p)

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(tensors), batch_size), desc="UNet infer"):
            batch_t = torch.stack(tensors[i : i + batch_size]).to(device)
            with autocast(enabled=(device == "cuda")):
                prob = model(batch_t)
            pred = (prob >= threshold).float().cpu().numpy()
            for j, m in enumerate(pred):
                mask = (m[0] * 255).astype(np.uint8)
                src = ok_paths[i + j]
                outputs[src] = mask
                out_name = Path(src).stem + "_mask.png"
                cv2.imwrite(str(out_dir / out_name), mask)

    return outputs


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Overlay segmentation mask on image."""
    overlay = img_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0)


def compare_segmentation_methods(config: dict, sample_n: int = 100) -> None:
    """Compare U-Net vs Otsu vs GrabCut using Otsu as reference."""
    img_dir = Path(config["data"]["isic_images_dir"])
    out_dir = PROJECT_ROOT / "outputs/seg_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([str(p) for p in img_dir.glob("*.jpg")])[:sample_n]
    if not image_paths:
        raise RuntimeError("No images found for comparison")

    method_scores = {"Otsu": [], "GrabCut": [], "U-Net": []}
    method_times = {"Otsu": 0.0, "GrabCut": 0.0, "U-Net": 0.0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ckpt = Path(config.get("preprocessing", {}).get("unet_checkpoint", "outputs/unet/best_unet.pth"))
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    start = time.time()
    unet_masks = predict_masks_batch(model, image_paths, device=device, batch_size=16, threshold=0.5)
    method_times["U-Net"] = len(image_paths) / max(time.time() - start, 1e-6)

    rows = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        ref = segment_otsu(img)

        t0 = time.time()
        otsu = segment_otsu(img)
        method_times["Otsu"] += time.time() - t0

        t0 = time.time()
        gc = segment_grabcut(img)
        method_times["GrabCut"] += time.time() - t0

        un = unet_masks.get(p, np.zeros_like(ref))

        def iou(a: np.ndarray, b: np.ndarray) -> float:
            ai = a > 0
            bi = b > 0
            inter = np.logical_and(ai, bi).sum()
            union = np.logical_or(ai, bi).sum()
            return float(inter / max(union, 1))

        def dice(a: np.ndarray, b: np.ndarray) -> float:
            ai = a > 0
            bi = b > 0
            inter = np.logical_and(ai, bi).sum()
            return float((2 * inter) / max(ai.sum() + bi.sum(), 1))

        for m_name, mask in [("Otsu", otsu), ("GrabCut", gc), ("U-Net", un)]:
            method_scores[m_name].append((iou(mask, ref), dice(mask, ref)))

        rows.append(
            {
                "image": Path(p).name,
                "otsu_iou": method_scores["Otsu"][-1][0],
                "grabcut_iou": method_scores["GrabCut"][-1][0],
                "unet_iou": method_scores["U-Net"][-1][0],
            }
        )

    method_times["Otsu"] = len(image_paths) / max(method_times["Otsu"], 1e-6)
    method_times["GrabCut"] = len(image_paths) / max(method_times["GrabCut"], 1e-6)

    pd.DataFrame(rows).to_csv(out_dir / "method_iou_table.csv", index=False)

    summary = []
    for m in ["Otsu", "GrabCut", "U-Net"]:
        ious = [x[0] for x in method_scores[m]]
        dices = [x[1] for x in method_scores[m]]
        summary.append({"Method": m, "Mean IoU": np.mean(ious), "Mean Dice": np.mean(dices), "Speed (img/s)": method_times[m]})
    sum_df = pd.DataFrame(summary)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(sum_df))
    plt.bar(x - 0.2, sum_df["Mean IoU"], width=0.2, label="IoU")
    plt.bar(x, sum_df["Mean Dice"], width=0.2, label="Dice")
    plt.bar(x + 0.2, sum_df["Speed (img/s)"] / max(sum_df["Speed (img/s)"].max(), 1), width=0.2, label="Speed (norm)")
    plt.xticks(x, sum_df["Method"].tolist())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "unet_vs_classical_bar.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_channels=3, n_classes=1).to(device)
    x = torch.randn(2, 3, 300, 300).to(device)
    out = model(x)
    assert out.shape == (2, 1, 300, 300), f"Wrong shape: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "Output not in [0,1]"
    print(f"U-Net output shape: {out.shape}  device: {out.device}  PASSED")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
