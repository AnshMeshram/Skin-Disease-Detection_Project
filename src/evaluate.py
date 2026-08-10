from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
        # If one-vs-rest has only one class present, ROC-AUC is undefined.
        if len(np.unique(y_bin)) < 2:
            aucs.append(float("nan"))
            continue
        aucs.append(float(roc_auc_score(y_bin, y_probs[:, i])))
    return np.asarray(aucs, dtype=np.float64)


def evaluate_model(model, dataloader, class_names):
    """Evaluate model on all 7 metrics and build per-class table."""
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    num_classes = len(class_names)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                raise ValueError("Dataloader must return at least (inputs, targets)")

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
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
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
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
        "accuracy": acc,
        "accuracy_pct": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "specificity_macro": spec_macro,
        "roc_auc_macro": auc_macro,
        "per_class": per_class,
    }

    _print_metrics_report(metrics, class_names)
    return metrics, y_true, y_pred, y_probs


def _print_metrics_report(metrics: dict, class_names) -> None:
    print("=" * 64)
    print("Metric                       Value")
    print("-" * 44)
    print(f"Accuracy                    {metrics['accuracy_pct']:.4f} %")
    print(f"Balanced Accuracy            {metrics['balanced_accuracy']:.6f}")
    print(f"F1 (macro)                   {metrics['f1_macro']:.6f}")
    print(f"Precision (macro)            {metrics['precision_macro']:.6f}")
    print(f"Recall (macro)               {metrics['recall_macro']:.6f}")
    print(f"Specificity (macro)          {metrics['specificity_macro']:.6f}")
    print(f"ROC-AUC (macro)              {metrics['roc_auc_macro']:.6f}")
    print("=" * 64)
    print()

    print("Class        Prec    Recall      F1    Spec     AUC")
    print("-" * 53)
    p_list, r_list, f_list, s_list, a_list = [], [], [], [], []
    for cls in class_names:
        row = metrics["per_class"][str(cls)]
        auc_val = row["roc_auc"] if row["roc_auc"] is not None else float("nan")
        p_list.append(row["precision"])
        r_list.append(row["recall"])
        f_list.append(row["f1"])
        s_list.append(row["specificity"])
        a_list.append(auc_val)
        print(
            f"{str(cls):<10}  {row['precision']:.4f}  {row['recall']:.4f}  "
            f"{row['f1']:.4f}  {row['specificity']:.4f}  {auc_val:.4f}"
        )

    print("-" * 53)
    print(
        f"MACRO       {np.nanmean(p_list):.4f}  {np.nanmean(r_list):.4f}  "
        f"{np.nanmean(f_list):.4f}  {np.nanmean(s_list):.4f}  {np.nanmean(a_list):.4f}"
    )


def save_results(metrics: dict, json_path: str | Path):
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # Lightweight self-check for module imports and save_results.
    demo = {
        "accuracy_pct": 96.12,
        "balanced_accuracy": 0.9412,
        "f1_macro": 0.9612,
        "precision_macro": 0.9587,
        "recall_macro": 0.9638,
        "specificity_macro": 0.9745,
        "roc_auc_macro": 0.9912,
        "per_class": {},
    }
    save_results(demo, "outputs/eval_demo.json")
    print("evaluate.py ready")
