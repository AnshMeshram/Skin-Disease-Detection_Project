from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import NUM_CLASSES, create_fold_dataloaders
from src.model_factory import build_model
from torch.amp import autocast, GradScaler


@dataclass(slots=True)
class TrainResult:
    best_bal_acc: float
    best_epoch: int
    run_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path


def print_checkpoint_status(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        print(f"[checkpoint] missing: {path}")
        return

    ckpt = torch.load(path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    print(f"[checkpoint] {path}")
    print(f"Model: {ckpt.get('model_name', '-')}")
    print(f"Fold: {ckpt.get('fold', '-')}")
    print(f"Epoch: {ckpt.get('epoch', '-')}")
    print(f"Phase: {ckpt.get('phase', '-')}")
    print(f"Best bal_acc: {ckpt.get('best_bal_acc', 0.0):.4f}")
    print(f"Last train_acc: {metrics.get('train_acc', 0.0):.2f}%")
    print(f"val_acc: {metrics.get('val_acc', 0.0):.2f}%")
    print(f"F1: {metrics.get('val_f1', 0.0):.4f}")
    print(f"Recall: {metrics.get('val_recall', 0.0):.4f}")
    print(f"Specificity: {metrics.get('val_specificity', 0.0):.4f}")


class Trainer:
    def __init__(self, model: torch.nn.Module, config: dict, class_weights, fold: int):
        self.model = model
        self.config = config
        self.fold = int(fold)
        self.epochs = int(config.get("epochs", 80))
        self.steps_per_epoch = int(config.get("steps_per_epoch", 0))
        self.epoch_time_target_sec = int(config.get("epoch_time_target_sec", 180))
        self.batch_size = int(config.get("batch_size", 32))
        self.learning_rate = float(config.get("learning_rate", 8e-5))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.plateau_patience = int(config.get("plateau_patience", 0))
        self.plateau_min_delta = float(config.get("plateau_min_delta", 0.002))
        self.resume_from_best = bool(config.get("resume_from_best", False))
        self.early_stopping_enabled = False  # disabled by request
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.use_amp = bool(config.get("amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.loss_name = str(config.get("loss", "cross_entropy")).lower()
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.label_smoothing = float(config.get("label_smoothing", 0.0))
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, int(config.get("scheduler_t0", 5))),
            T_mult=max(1, int(config.get("scheduler_tmult", 2))),
            eta_min=float(config.get("min_lr", 1e-7)),
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_balanced_accuracy": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_specificity": [],
            "lr": [],
        }
        self.best_bal_acc = -float("inf")
        self.best_epoch = 0
        self.phase = "phase1"

        if hasattr(self.model, "freeze_backbone") and bool(config.get("freeze_backbone", False)):
            try:
                self.model.freeze_backbone()
            except Exception:
                pass

    def _criterion(self) -> torch.nn.Module:
        if self.loss_name == "focal":
            from src.losses import FocalLoss

            return FocalLoss(gamma=self.focal_gamma, alpha=self.class_weights)
        if self.label_smoothing > 0:
            return torch.nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        return torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def _get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_phase(self, epoch: int) -> None:
        phase1_epochs = int(self.config.get("phase1_epochs", 0))
        phase2_start = int(self.config.get("phase2_start_epoch", phase1_epochs + 1))
        phase3_start = int(self.config.get("phase3_start_epoch", self.epochs + 1))

        if epoch < phase2_start:
            self.phase = "phase1"
            if hasattr(self.model, "freeze_backbone"):
                try:
                    self.model.freeze_backbone()
                except Exception:
                    pass
        elif epoch < phase3_start:
            self.phase = "phase2"
            if hasattr(self.model, "unfreeze_backbone"):
                try:
                    self.model.unfreeze_backbone()
                except Exception:
                    pass
        else:
            self.phase = "phase3"
            if hasattr(self.model, "unfreeze_backbone"):
                try:
                    self.model.unfreeze_backbone()
                except Exception:
                    pass

    def _save_checkpoint(self, path: Path, epoch: int, model_name: str, metrics: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model_name": model_name,
            "fold": self.fold,
            "phase": self.phase,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "best_bal_acc": self.best_bal_acc,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, path)

    def _load_model_only(self, ckpt_path: Path) -> int:
        if not ckpt_path.exists():
            return 1
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state)
        self.best_bal_acc = float(ckpt.get("best_bal_acc", self.best_bal_acc))
        self.best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
        self.phase = str(ckpt.get("phase", "phase1"))
        return int(ckpt.get("epoch", 0)) + 1

    def _load_latest(self, latest_path: Path) -> int:
        if not latest_path.exists():
            return 1
        ckpt = torch.load(latest_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            pass
        try:
            if self.scheduler is not None and "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
        if self.scaler is not None and ckpt.get("scaler_state") is not None:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass
        self.best_bal_acc = float(ckpt.get("best_bal_acc", -float("inf")))
        self.best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
        self.phase = str(ckpt.get("phase", "phase1"))
        return int(ckpt.get("epoch", 0)) + 1

    def _save_history(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if len(self.history["train_loss"]) == 0:
            return

        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history["train_loss"], label="train_loss")
        plt.plot(epochs, self.history["val_loss"], label="val_loss")
        plt.plot(epochs, self.history["val_balanced_accuracy"], label="val_bal_acc")
        plt.plot(epochs, self.history["val_f1"], label="val_f1")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "training_curves.png", dpi=300)
        plt.close()

    def _specificity_macro(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> float:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        spec_per_class = []
        total = cm.sum()
        for i in range(num_classes):
            tp = float(cm[i, i])
            fp = float(cm[:, i].sum() - tp)
            fn = float(cm[i, :].sum() - tp)
            tn = float(total - tp - fp - fn)
            denom = tn + fp
            spec_per_class.append(tn / denom if denom > 0 else 0.0)
        return float(np.mean(spec_per_class)) if spec_per_class else 0.0

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, loss_value: float) -> dict[str, float]:
        return {
            "loss": float(loss_value),
            "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "specificity_macro": float(self._specificity_macro(y_true, y_pred, num_classes=NUM_CLASSES)),
        }

    def train_epoch(self, dl: DataLoader, epoch: int | None = None) -> tuple[float, float]:
        self.model.train()
        criterion = self._criterion()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        steps = 0

        pbar = tqdm(dl, desc=f"train {epoch}" if epoch is not None else "train", leave=False)
        for batch in pbar:
            if isinstance(batch, dict):
                images = batch["image"]
                targets = batch["target"]
            else:
                images, targets = batch[:2]

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                with autocast('cuda'):
                    logits = self.model(images)
                    loss = criterion(logits, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = criterion(logits, targets)
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                try:
                    self.scheduler.step(epoch - 1 + (steps + 1) / max(1, len(dl)) if epoch is not None else None)
                except TypeError:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

            total_loss += float(loss.item())
            preds = torch.argmax(logits.detach(), dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            steps += 1
            # Show running metrics in progress bar
            running_loss = total_loss / max(1, steps)
            running_acc = float(accuracy_score(all_targets, all_preds) * 100.0)
            running_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
            running_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
            running_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
            pbar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.2f}",
                "f1": f"{running_f1:.3f}",
                "prec": f"{running_prec:.3f}",
                "rec": f"{running_rec:.3f}"
            })

        train_loss = total_loss / max(1, steps)
        train_acc = float(accuracy_score(all_targets, all_preds) * 100.0)
        return train_loss, train_acc

    @torch.no_grad()
    def val_epoch(self, dl: DataLoader, epoch: int | None = None) -> dict[str, float]:
        self.model.eval()
        criterion = self._criterion()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        steps = 0

        pbar = tqdm(dl, desc=f"val {epoch}" if epoch is not None else "val", leave=False)
        for batch in pbar:
            if isinstance(batch, dict):
                images = batch["image"]
                targets = batch["target"]
            else:
                images, targets = batch[:2]

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).long()
            logits = self.model(images)
            loss = criterion(logits, targets)

            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            steps += 1
            # Show running metrics in progress bar
            running_loss = total_loss / max(1, steps)
            running_acc = float(accuracy_score(all_targets, all_preds) * 100.0)
            running_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
            running_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
            running_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
            pbar.set_postfix({
                "loss": f"{running_loss:.4f}",
                "acc": f"{running_acc:.2f}",
                "f1": f"{running_f1:.3f}",
                "prec": f"{running_prec:.3f}",
                "rec": f"{running_rec:.3f}"
            })

        avg_loss = total_loss / max(1, steps)
        return self._compute_metrics(np.asarray(all_targets), np.asarray(all_preds), avg_loss)

    def _should_unfreeze(self) -> bool:
        series = self.history.get("val_balanced_accuracy", [])
        if len(series) < self.plateau_patience + 1:
            return False
        recent = series[-(self.plateau_patience + 1):]
        best_prev = max(recent[:-1])
        return recent[-1] < best_prev + self.plateau_min_delta

    def train(self, tr_dl: DataLoader, val_dl: DataLoader, model_name: str, resume: bool = False) -> float:
        run_dir = Path("outputs") / model_name / f"fold_{self.fold}"
        latest_path = run_dir / "latest.pth"
        best_path = run_dir / "best.pth"
        run_dir.mkdir(parents=True, exist_ok=True)

        if resume and self.resume_from_best:
            start_epoch = self._load_model_only(best_path)
            print(f"Resuming from best checkpoint (fresh optimizer) at epoch {start_epoch}: {best_path}")
        else:
            start_epoch = self._load_latest(latest_path) if resume else 1
            if resume and latest_path.exists():
                print(f"Resuming from latest checkpoint at epoch {start_epoch}: {latest_path}")

        if start_epoch <= 1 and not resume:
            self.best_bal_acc = -float("inf")
            self.best_epoch = 0

        criterion = self._criterion()
        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.time()
            self._set_phase(epoch)

            tr_loss, tr_acc = self.train_epoch(tr_dl, epoch=epoch)
            val = self.val_epoch(val_dl, epoch=epoch)

            lr = self._get_lr()
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val["loss"])
            self.history["val_acc"].append(val["accuracy"])
            self.history["val_balanced_accuracy"].append(val["balanced_acc"])
            self.history["val_f1"].append(val["f1_macro"])
            self.history["val_precision"].append(val["precision_macro"])
            self.history["val_recall"].append(val["recall_macro"])
            self.history["val_specificity"].append(val["specificity_macro"])
            self.history["lr"].append(lr)

            metrics_payload = {
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val["loss"],
                "val_acc": val["accuracy"],
                "val_balanced_accuracy": val["balanced_acc"],
                "val_f1": val["f1_macro"],
                "val_precision": val["precision_macro"],
                "val_recall": val["recall_macro"],
                "val_specificity": val["specificity_macro"],
                "lr": lr,
            }

            self._save_checkpoint(latest_path, epoch, model_name, metrics_payload)

            elapsed = int(time.time() - epoch_start)
            print(
                f"Epoch {epoch}/{self.epochs} [{self.phase}] {elapsed}s | "
                f"train_loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
                f"val_loss {val['loss']:.4f} acc {val['accuracy']:.2f}% bal {val['balanced_acc']:.4f} | "
                f"f1 {val['f1_macro']:.4f} precision {val['precision_macro']:.4f} "
                f"recall {val['recall_macro']:.4f} spec {val['specificity_macro']:.4f} | lr {lr:.6e}"
            )

            if val["balanced_acc"] > self.best_bal_acc + 1e-12:
                self.best_bal_acc = float(val["balanced_acc"])
                self.best_epoch = int(epoch)
                self._save_checkpoint(best_path, epoch, model_name, metrics_payload)

            self._save_history(run_dir)

            if self.early_stopping_enabled and self._should_unfreeze():
                print("Plateau detected but early stopping is disabled; continuing training.")
                if hasattr(self.model, "unfreeze_backbone"):
                    try:
                        self.model.unfreeze_backbone()
                    except Exception:
                        pass

        return self.best_bal_acc


def _resolve_model_name(config: dict, model_name: str | None = None) -> str:
    if model_name:
        return str(model_name)
    if "model" in config:
        return str(config["model"])
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return "inception_v3"


def _build_class_weights(df) -> torch.Tensor | None:
    if df is None or "label" not in df.columns:
        return None
    counts = df["label"].value_counts().sort_index()
    weights = counts.sum() / (len(counts) * counts.clip(lower=1))
    return torch.as_tensor(weights.values, dtype=torch.float32)


def train_all_folds(config: dict, df, preprocess_fn: Any = None, resume: bool = False, fold_ids: list[int] | None = None, warm_start: str = ""):
    model_names = config.get("architectures") or [config.get("model", "efficientnet_b3")]
    model_name = _resolve_model_name(config)

    if fold_ids is None:
        n_folds = int(config.get("n_folds", config.get("num_folds", 5)))
        fold_ids = list(range(n_folds))

    results: dict[str, dict[int, float]] = {model_name: {}}

    for fold in fold_ids:
        tr_dl, val_dl, class_weights = create_fold_dataloaders(
            df,
            config=config,
            fold_idx=fold,
            preprocess_fn=preprocess_fn,
        )

        model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=bool(config.get("pretrained", True)))
        trainer = Trainer(model=model, config=config, class_weights=class_weights, fold=fold)
        best_bal_acc = trainer.train(tr_dl, val_dl, model_name=model_name, resume=resume)
        results[model_name][fold] = float(best_bal_acc)

    return results
