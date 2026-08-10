from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset import CLASS_NAMES, build_dataframe, create_fold_dataloaders
from src.model_factory import build_model


def _specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specs = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else 0.0)
    return np.asarray(specs, dtype=np.float64)


def _auc_per_class(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int) -> np.ndarray:
    aucs = []
    for i in range(num_classes):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            aucs.append(float("nan"))
            continue
        aucs.append(float(roc_auc_score(y_bin, y_probs[:, i])))
    return np.asarray(aucs, dtype=np.float64)


def _normalize_weights(weights: Iterable[float], n_models: int) -> list[float]:
    arr = np.asarray(list(weights), dtype=np.float64)
    if arr.size != n_models:
        raise ValueError(f"weights length must be {n_models}, got {arr.size}")
    arr = np.clip(arr, 1e-8, None)
    arr = arr / arr.sum()
    return arr.tolist()


class EnsembleModel:
    def __init__(self, models: list, weights: list = None, temperatures: list = None):
        if not models:
            raise ValueError("models list is empty")
        self.models = list(models)
        self.n_models = len(self.models)

        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        self.weights = _normalize_weights(weights, self.n_models)

        if temperatures is None:
            temperatures = [1.0] * self.n_models
        if len(temperatures) != self.n_models:
            raise ValueError(f"temperatures length must be {self.n_models}, got {len(temperatures)}")
        self.temperatures = [max(0.05, float(t)) for t in temperatures]

        self.device = next(self.models[0].parameters()).device
        for m in self.models:
            m.eval()

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        probs_sum = None
        with torch.no_grad():
            for model, w, t in zip(self.models, self.weights, self.temperatures):
                logits = model(inputs)
                probs = F.softmax(logits / t, dim=1)
                if probs_sum is None:
                    probs_sum = w * probs
                else:
                    probs_sum = probs_sum + w * probs

        if probs_sum is None:
            raise RuntimeError("No model predictions produced")

        row_sum = probs_sum.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs_sum = probs_sum / row_sum
        return probs_sum

    def predict_with_confidence(self, inputs) -> tuple:
        probs = self.predict(inputs)
        conf, cls_idx = probs.max(dim=1)
        return cls_idx, conf, probs


def _candidate_checkpoint_paths(config: dict, arch: str, fold: int) -> list[Path]:
    out_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    return [
        out_root / arch / f"fold_{fold}" / "best.pth",
        out_root / "models" / arch / f"fold_{fold}" / "best.pth",
    ]


def _checkpoint_bal_acc(path: Path) -> float:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "best_bal_acc" in ckpt:
            return float(ckpt.get("best_bal_acc", 0.0))
        metrics = ckpt.get("metrics", {})
        if isinstance(metrics, dict) and "balanced_acc" in metrics:
            return float(metrics.get("balanced_acc", 0.0))
        if isinstance(metrics, dict) and "val_balanced_accuracy" in metrics:
            return float(metrics.get("val_balanced_accuracy", 0.0))
    return 0.0


def load_best_checkpoints(config, architectures) -> list:
    folds = int(config.get("folds", 5))
    num_classes = int(config.get("num_classes", 9))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = []
    for arch in architectures:
        best_path = None
        best_bal = -float("inf")
        for fold in range(folds):
            for cand in _candidate_checkpoint_paths(config, arch, fold):
                if not cand.exists():
                    continue
                bal = _checkpoint_bal_acc(cand)
                if bal > best_bal:
                    best_bal = bal
                    best_path = cand

        if best_path is None:
            raise FileNotFoundError(f"No best checkpoint found for architecture {arch}")

        model = build_model(arch, num_classes=num_classes, pretrained=False).to(device)
        ckpt = torch.load(best_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()
        loaded.append((model, arch))
        print(f"Loaded {arch} from {best_path} (best bal_acc={best_bal:.4f})")

    return loaded


def _collect_logits_labels(model, val_loader):
    model.eval()
    device = next(model.parameters()).device
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def temperature_calibration(model, val_loader) -> float:
    logits, labels = _collect_logits_labels(model, val_loader)

    def nll(temp: float) -> float:
        t = max(0.05, float(temp))
        loss = F.cross_entropy(logits / t, labels, reduction="mean")
        return float(loss.item())

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded")
        if res.success:
            return float(res.x)
    except Exception:
        pass

    best_t = 1.0
    best_loss = float("inf")
    for t in np.arange(0.5, 3.01, 0.05):
        loss = nll(float(t))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def _collect_model_probs(models: list, val_loader, temperatures: list[float] | None = None):
    probs_per_model = []
    y_true = None
    for i, model in enumerate(models):
        logits, labels = _collect_logits_labels(model, val_loader)
        t = 1.0 if temperatures is None else float(temperatures[i])
        probs = F.softmax(logits / max(0.05, t), dim=1).numpy()
        probs_per_model.append(probs)
        if y_true is None:
            y_true = labels.numpy()
    return probs_per_model, y_true


def _balanced_accuracy_from_probs(y_true: np.ndarray, probs: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float(balanced_accuracy_score(y_true, preds))


def _weight_grid_search(probs_per_model: list[np.ndarray], y_true: np.ndarray) -> list[float]:
    n_models = len(probs_per_model)
    if n_models == 1:
        return [1.0]

    values = [i / 10.0 for i in range(1, 10)]
    best_w = [1.0 / n_models] * n_models
    best_score = -float("inf")

    for combo in itertools.product(values, repeat=n_models):
        s = float(sum(combo))
        if abs(s - 1.0) > 1e-9:
            continue
        probs = np.zeros_like(probs_per_model[0], dtype=np.float64)
        for w, p in zip(combo, probs_per_model):
            probs += w * p
        score = _balanced_accuracy_from_probs(y_true, probs)
        if score > best_score:
            best_score = score
            best_w = list(combo)

    return _normalize_weights(best_w, n_models)


def optimize_weights(models, val_loader, class_names) -> list:
    _ = class_names
    bare_models = [m[0] if isinstance(m, tuple) else m for m in models]
    probs_per_model, y_true = _collect_model_probs(bare_models, val_loader)

    try:
        from scipy.optimize import minimize

        n_models = len(bare_models)
        x0 = np.ones(n_models, dtype=np.float64) / n_models
        bounds = [(0.05, 1.0)] * n_models
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)

        def obj(x):
            x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, None)
            x = x / x.sum()
            probs = np.zeros_like(probs_per_model[0], dtype=np.float64)
            for w, p in zip(x, probs_per_model):
                probs += w * p
            return -_balanced_accuracy_from_probs(y_true, probs)

        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            return _normalize_weights(res.x.tolist(), n_models)
    except Exception:
        pass

    return _weight_grid_search(probs_per_model, y_true)


def evaluate_ensemble(ensemble, dataloader, class_names) -> dict:
    num_classes = len(class_names)
    y_true = []
    y_pred = []
    y_probs = []

    device = ensemble.device

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            probs = ensemble.predict(inputs)
            preds = probs.argmax(dim=1)

            y_true.extend(targets.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_probs.extend(probs.detach().cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    spec_per_cls = _specificity_per_class(y_true, y_pred, num_classes=num_classes)
    spec_macro = float(np.nanmean(spec_per_cls))

    auc_per_cls = _auc_per_class(y_true, y_probs, num_classes=num_classes)
    auc_macro = float(np.nanmean(auc_per_cls))

    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[str(cls)] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f_cls[i]),
            "specificity": float(spec_per_cls[i]),
            "roc_auc": float(auc_per_cls[i]) if np.isfinite(auc_per_cls[i]) else None,
        }

    metrics = {
        "accuracy_pct": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "specificity_macro": spec_macro,
        "roc_auc_macro": auc_macro,
        "per_class": per_class,
    }

    print("=" * 72)
    print("Ensemble Metrics")
    print("-" * 72)
    print(f"Accuracy: {metrics['accuracy_pct']:.4f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"F1 (macro): {metrics['f1_macro']:.6f}")
    print(f"Precision (macro): {metrics['precision_macro']:.6f}")
    print(f"Recall (macro): {metrics['recall_macro']:.6f}")
    print(f"Specificity (macro): {metrics['specificity_macro']:.6f}")
    print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.6f}")
    print("=" * 72)

    print("Class        Prec    Recall      F1    Spec     AUC")
    print("-" * 56)
    for cls in class_names:
        row = per_class[str(cls)]
        auc_val = row["roc_auc"] if row["roc_auc"] is not None else float("nan")
        print(
            f"{str(cls):<10}  {row['precision']:.4f}  {row['recall']:.4f}  "
            f"{row['f1']:.4f}  {row['specificity']:.4f}  {auc_val:.4f}"
        )

    return metrics


def run_ensemble_stage(config: dict, architectures: list[str] | None = None, fold_idx: int = 0) -> dict:
    if architectures is None:
        architectures = config.get("architectures")
        if not architectures:
            architectures = ["efficientnet_b3", "inception_v3", "convnext_tiny"]
        if isinstance(architectures, str):
            architectures = [architectures]

    prefer_pre = bool(config.get("preprocessing", {}).get("use_preprocessed", True))
    df = build_dataframe(config, prefer_preprocessed=prefer_pre)
    _tr_dl, val_loader, _cw = create_fold_dataloaders(df, config, fold_idx=fold_idx, preprocess_fn=None)

    loaded = load_best_checkpoints(config, architectures)
    models = [m for m, _name in loaded]
    arch_names = [n for _m, n in loaded]

    temperatures = []
    for model, name in loaded:
        t = temperature_calibration(model, val_loader)
        temperatures.append(float(t))
        print(f"Calibrated temperature for {name}: {t:.3f}")

    weights = optimize_weights(models, val_loader, CLASS_NAMES)
    print(f"Optimized weights: {weights}")

    ensemble = EnsembleModel(models=models, weights=weights, temperatures=temperatures)
    metrics = evaluate_ensemble(ensemble, val_loader, CLASS_NAMES)

    out_root = Path(config.get("data", {}).get("output_dir", "outputs")) / "ensemble"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "ensemble_metrics.json"

    payload = {
        "architectures": arch_names,
        "weights": weights,
        "temperatures": temperatures,
        "metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    title = " + ".join([n.replace("_", " ").title() for n in arch_names])
    print()
    print(f"Ensemble ({title})")
    print(
        f"Accuracy: {metrics['accuracy_pct']:.1f}% | "
        f"F1: {metrics['f1_macro']:.3f} | "
        f"AUC: {metrics['roc_auc_macro']:.3f}"
    )
    print(f"Saved: {out_path}")

    return payload


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_ensemble_stage(cfg)
