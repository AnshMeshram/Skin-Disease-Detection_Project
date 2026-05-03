from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from src.dataset import CLASS_NAMES, build_dataframe, create_fold_dataloaders
from src.evaluate import evaluate_model, save_results
from src.model_factory import build_model
from src.plots import plot_fold_summary, save_all_fold_plots, save_metrics_txt


def _load_fold_history(out_root: Path, model_name: str, fold: int, phase1_default: int) -> dict[str, Any]:
    path = out_root / model_name / f"fold_{fold}" / "history.json"
    if not path.exists():
        return {"phase1_epochs": phase1_default}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"phase1_epochs": phase1_default}
    data["phase1_epochs"] = int(data.get("phase1_epochs", phase1_default))
    return data


def _save_figure(fig: plt.Figure, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {label}: {out_path}")


def _plot_cv_training_curves(histories: dict[int, dict[str, Any]], model_name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_train_acc, ax_val_acc = axes[0]
    ax_train_loss, ax_val_loss = axes[1]

    max_train_acc = 1.0
    max_val_acc = 1.0
    max_train_loss = 1.0
    max_val_loss = 1.0

    for fold in sorted(histories.keys()):
        h = histories[fold]
        train_acc = np.asarray(h.get("train_acc", []), dtype=float)
        val_acc = np.asarray(h.get("val_acc", []), dtype=float)
        train_loss = np.asarray(h.get("train_loss", []), dtype=float)
        val_loss = np.asarray(h.get("val_loss", []), dtype=float)
        epochs = np.arange(1, len(train_acc) + 1)

        if len(train_acc) > 0:
            ax_train_acc.plot(epochs, train_acc, label=f"Fold {fold}")
            max_train_acc = max(max_train_acc, float(np.max(train_acc)))

        if len(val_acc) > 0:
            ax_val_acc.plot(np.arange(1, len(val_acc) + 1), val_acc, label=f"Fold {fold}")
            best_i = int(np.argmax(val_acc))
            ax_val_acc.scatter(best_i + 1, float(val_acc[best_i]), s=20)
            max_val_acc = max(max_val_acc, float(np.max(val_acc)))

        if len(train_loss) > 0:
            ax_train_loss.plot(np.arange(1, len(train_loss) + 1), train_loss, label=f"Fold {fold}")
            max_train_loss = max(max_train_loss, float(np.max(train_loss)))

        if len(val_loss) > 0:
            ax_val_loss.plot(np.arange(1, len(val_loss) + 1), val_loss, label=f"Fold {fold}")
            max_val_loss = max(max_val_loss, float(np.max(val_loss)))

    ax_train_acc.set_title("Train Accuracy by Fold")
    ax_train_acc.set_xlabel("Epoch")
    ax_train_acc.set_ylabel("Accuracy (%)")
    ax_train_acc.set_ylim(0.0, max_train_acc * 1.02)
    ax_train_acc.grid(alpha=0.25)
    ax_train_acc.legend()

    ax_val_acc.set_title("Validation Accuracy by Fold")
    ax_val_acc.set_xlabel("Epoch")
    ax_val_acc.set_ylabel("Accuracy (%)")
    ax_val_acc.set_ylim(0.0, max_val_acc * 1.02)
    ax_val_acc.grid(alpha=0.25)
    ax_val_acc.legend()

    ax_train_loss.set_title("Train Loss by Fold")
    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Loss")
    ax_train_loss.set_ylim(0.0, max_train_loss * 1.05)
    ax_train_loss.grid(alpha=0.25)
    ax_train_loss.legend()

    ax_val_loss.set_title("Validation Loss by Fold")
    ax_val_loss.set_xlabel("Epoch")
    ax_val_loss.set_ylabel("Loss")
    ax_val_loss.set_ylim(0.0, max_val_loss * 1.05)
    ax_val_loss.grid(alpha=0.25)
    ax_val_loss.legend()

    fig.suptitle(f"{model_name} - 5-Fold Detailed Training Curves", fontsize=16)
    _save_figure(fig, out_dir / f"{model_name}_cv_training_curves.png", "combined CV training curves")


def _plot_combined_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path, model_name: str) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Combined Confusion Matrix (All Folds, Counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)

    _save_figure(fig, out_dir / f"{model_name}_combined_confusion_matrix.png", "combined confusion matrix")

    cmf = cm.astype(np.float64)
    row_sum = cmf.sum(axis=1, keepdims=True)
    cmn = np.divide(cmf, np.where(row_sum == 0, 1.0, row_sum)) * 100.0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Combined Confusion Matrix (All Folds, Row-normalized %)")

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.1f}", ha="center", va="center", fontsize=8)

    _save_figure(fig, out_dir / f"{model_name}_combined_confusion_matrix_norm.png", "combined normalized confusion matrix")


def _plot_combined_roc_pr(y_true: np.ndarray, y_probs: np.ndarray, class_names: list[str], out_dir: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    aucs = []
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        c_auc = auc(fpr, tpr)
        aucs.append(c_auc)
        ax.plot(fpr, tpr, label=f"{cname} AUC={c_auc:.4f}")
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Combined ROC Curves (All Folds, macro AUC={macro_auc:.4f})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, out_dir / f"{model_name}_combined_roc_curves.png", "combined ROC curves")

    fig, ax = plt.subplots(figsize=(10, 8))
    pr_aucs = []
    for i, cname in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        ax.plot(recall, precision, label=f"{cname} AUC={pr_auc:.4f}")
    macro_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Combined Precision-Recall Curves (All Folds, macro AUC={macro_pr_auc:.4f})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, out_dir / f"{model_name}_combined_precision_recall_curves.png", "combined precision-recall curves")


def _plot_per_class_cv_stats(fold_metrics: list[dict[str, Any]], class_names: list[str], out_dir: Path, model_name: str) -> None:
    metrics_keys = ["precision", "recall", "f1", "specificity"]

    class_to_values = {c: {k: [] for k in metrics_keys} for c in class_names}
    for row in fold_metrics:
        per_class = row["metrics"].get("per_class", {})
        for cname in class_names:
            vals = per_class.get(cname, {})
            for k in metrics_keys:
                class_to_values[cname][k].append(float(vals.get(k, 0.0)))

    x = np.arange(len(class_names))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(15, 6))
    for k, off in zip(metrics_keys, offsets):
        means = [float(np.mean(class_to_values[c][k])) for c in class_names]
        stds = [float(np.std(class_to_values[c][k])) for c in class_names]
        ax.bar(x + off, means, width, yerr=stds, capsize=3, label=f"{k} mean±std")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics Across Folds (mean ± std)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    _save_figure(fig, out_dir / f"{model_name}_per_class_cv_stats.png", "per-class CV metric stats")


def _write_cv_tables(fold_metrics: list[dict[str, Any]], out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in fold_metrics:
        m = row["metrics"]
        rows.append(
            {
                "fold": row["fold"],
                "accuracy_pct": float(m.get("accuracy_pct", 0.0)),
                "balanced_accuracy": float(m.get("balanced_accuracy", 0.0)),
                "f1_macro": float(m.get("f1_macro", 0.0)),
                "precision_macro": float(m.get("precision_macro", 0.0)),
                "recall_macro": float(m.get("recall_macro", 0.0)),
                "specificity_macro": float(m.get("specificity_macro", 0.0)),
                "roc_auc_macro": float(m.get("roc_auc_macro", 0.0)),
            }
        )

    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys() if k != "fold"}
    stds = {k: float(np.std([r[k] for r in rows])) for k in rows[0].keys() if k != "fold"}
    best_row = max(rows, key=lambda r: r["accuracy_pct"])

    summary = {
        "model": model_name,
        "folds": [r["fold"] for r in rows],
        "rows": rows,
        "mean": means,
        "std": stds,
        "best_by_accuracy": best_row,
    }

    with open(out_dir / f"{model_name}_cv_detailed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / f"{model_name}_cv_detailed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow({})
        writer.writerow({"fold": "mean", **means})
        writer.writerow({"fold": "std", **stds})

    print(f"Saved summary JSON: {out_dir / f'{model_name}_cv_detailed_summary.json'}")
    print(f"Saved summary CSV: {csv_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed plots for folds 0..N-1")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--folds", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_folds(folds_arg: str, folds_total: int) -> list[int]:
    if not folds_arg.strip():
        return list(range(folds_total))
    vals = []
    for chunk in folds_arg.split(","):
        idx = int(chunk.strip())
        if idx < 0 or idx >= folds_total:
            raise ValueError(f"fold index out of range: {idx}")
        vals.append(idx)
    return vals


def main() -> None:
    args = _parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = args.model.strip() or str(config.get("model", "efficientnet_b3"))
    folds_total = int(config.get("folds", 5))
    fold_ids = _resolve_folds(args.folds, folds_total)

    out_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    eval_config = dict(config)
    eval_config["num_workers"] = 0
    eval_config["val_num_workers"] = 0
    eval_config["persistent_workers"] = False
    eval_config["prefetch_factor"] = 2
    eval_config["pin_memory"] = bool(torch.cuda.is_available())
    eval_config["batch_size"] = int(min(int(config.get("batch_size", 16)), 8))

    prefer_pre = bool(config.get("preprocessing", {}).get("use_preprocessed", True))
    df = build_dataframe(config, prefer_preprocessed=prefer_pre)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Evaluating folds: {fold_ids}")

    fold_metrics: list[dict[str, Any]] = []
    histories: dict[int, dict[str, Any]] = {}
    all_true = []
    all_pred = []
    all_probs = []

    num_classes = int(config.get("num_classes", len(CLASS_NAMES)))
    class_names = CLASS_NAMES[:num_classes]

    for fidx in fold_ids:
        ckpt_path = out_root / model_name / f"fold_{fidx}" / "best.pth"
        if not ckpt_path.exists():
            print(f"Skipping fold {fidx}: missing checkpoint {ckpt_path}")
            continue

        print(f"\n=== Evaluating fold {fidx} ===")
        _tr_dl, val_dl, _cw = create_fold_dataloaders(df, eval_config, fold_idx=fidx, preprocess_fn=None)

        model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)

        metrics, y_true, y_pred, y_probs = evaluate_model(model, val_dl, class_names)
        fold_metrics.append({"fold": fidx, "metrics": metrics})

        fold_out = out_root / "models" / model_name / f"fold_{fidx}"
        save_results(metrics, fold_out / "metrics.json")
        save_metrics_txt(metrics, fold_out / "metrics.txt")

        history = _load_fold_history(out_root, model_name, fidx, int(config.get("phase1_epochs", 0)))
        histories[fidx] = history

        save_all_fold_plots(
            history=history,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            per_class=metrics.get("per_class", {}),
            model_name=model_name,
            fold=fidx,
            out_dir=out_root,
        )

        all_true.append(np.asarray(y_true, dtype=np.int64))
        all_pred.append(np.asarray(y_pred, dtype=np.int64))
        all_probs.append(np.asarray(y_probs, dtype=np.float64))

    if not fold_metrics:
        raise RuntimeError("No folds were evaluated. Check checkpoint paths and fold indices.")

    ordered = sorted(fold_metrics, key=lambda x: int(x["fold"]))
    plot_fold_summary([r["metrics"] for r in ordered], model_name, out_root)

    cv_dir = out_root / "models" / model_name / "cv_detailed_plots"
    _plot_cv_training_curves(histories, model_name, cv_dir)

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    y_probs_all = np.concatenate(all_probs, axis=0)

    _plot_combined_confusion(y_true_all, y_pred_all, class_names, cv_dir, model_name)
    _plot_combined_roc_pr(y_true_all, y_probs_all, class_names, cv_dir, model_name)
    _plot_per_class_cv_stats(ordered, class_names, cv_dir, model_name)
    _write_cv_tables(ordered, cv_dir, model_name)

    print("\nDONE: detailed fold plots and combined CV plots generated.")


if __name__ == "__main__":
    main()
