from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_path: Path, label: str) -> None:
    _ensure_parent(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {label}: {out_path}")


def _default_class_names(n: int = 9) -> list[str]:
    names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]
    return names[:n]


def _safe_get(history: dict, key: str, default: float = 0.0) -> list[float]:
    vals = history.get(key, [])
    if isinstance(vals, list):
        return vals
    return [default]


def _fold_plot_dir(model_name: str, fold: int, out_dir: str | Path) -> Path:
    return Path(out_dir) / "models" / model_name / f"fold_{fold}" / "plots"


def plot_training_history(history, model_name, fold, out_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()

    train_loss = _safe_get(history, "train_loss")
    val_loss = _safe_get(history, "val_loss")
    train_acc = _safe_get(history, "train_acc")
    val_acc = _safe_get(history, "val_acc")
    epochs = np.arange(1, len(train_loss) + 1)

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    acc_vals = [float(v) for v in (train_acc + val_acc)]
    acc_top = max(acc_vals) * 1.02 if acc_vals else 100.0
    axes[1].set_ylim(0.0, max(1.0, acc_top))
    axes[1].legend()

    axes[2].plot(epochs, _safe_get(history, "val_balanced_accuracy"), color="tab:green")
    axes[2].set_title("Balanced Accuracy")
    axes[2].set_ylim(0.0, 1.0)

    axes[3].plot(epochs, _safe_get(history, "val_f1"), label="F1")
    axes[3].plot(epochs, _safe_get(history, "val_precision"), label="Precision")
    axes[3].plot(epochs, _safe_get(history, "val_recall"), label="Recall")
    axes[3].set_title("F1 / Precision / Recall")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].legend()

    axes[4].plot(epochs, _safe_get(history, "val_specificity"), color="tab:purple")
    axes[4].set_title("Specificity")
    axes[4].set_ylim(0.0, 1.0)

    axes[5].plot(epochs, _safe_get(history, "lr"), color="tab:red")
    axes[5].set_title("Learning Rate")
    axes[5].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_training_history.png"
    _save(fig, out_path, "training history")


def plot_confusion_matrix(y_true, y_pred, model_name, fold, out_dir, class_names):
    class_names = class_names or _default_class_names()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Counts)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_confusion_matrix.png"
    _save(fig, out_path, "confusion matrix")


def plot_confusion_matrix_norm(y_true, y_pred, model_name, fold, out_dir, class_names):
    class_names = class_names or _default_class_names()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, np.where(row_sums == 0, 1.0, row_sums)) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Row-normalized %)")

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_confusion_matrix_norm.png"
    _save(fig, out_path, "normalized confusion matrix")


def plot_roc_curves(y_true, y_probs, model_name, fold, out_dir, class_names):
    class_names = class_names or _default_class_names()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    macro_aucs = []

    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        class_auc = auc(fpr, tpr)
        macro_aucs.append(class_auc)
        ax.plot(fpr, tpr, label=f"{cls} AUC={class_auc:.4f}")

    macro_auc = float(np.mean(macro_aucs)) if macro_aucs else 0.0
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.axhline(0.99, color="red", linestyle="--", linewidth=1.2, label="Target 0.99")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (macro AUC={macro_auc:.4f})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_roc_curves.png"
    _save(fig, out_path, "ROC curves")


def plot_per_class_metrics(per_class, model_name, fold, out_dir):
    classes = list(per_class.keys())
    prec = [per_class[c].get("precision", 0.0) for c in classes]
    rec = [per_class[c].get("recall", 0.0) for c in classes]
    f1 = [per_class[c].get("f1", 0.0) for c in classes]
    spec = [per_class[c].get("specificity", 0.0) for c in classes]

    x = np.arange(len(classes))
    w = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5 * w, prec, w, label="Precision")
    ax.bar(x - 0.5 * w, rec, w, label="Recall")
    ax.bar(x + 0.5 * w, f1, w, label="F1")
    ax.bar(x + 1.5 * w, spec, w, label="Specificity")
    ax.axhline(0.96, color="red", linestyle="--", linewidth=1.2, label="Target 0.96")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-class metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_per_class_metrics.png"
    _save(fig, out_path, "per-class metrics")


def plot_class_distribution_bar(class_counts, title, out_path):
    classes = list(class_counts.keys())
    counts = np.asarray(list(class_counts.values()), dtype=int)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, counts, color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(v)), ha="center", va="bottom", fontsize=8)

    _save(fig, Path(out_path), "class distribution bar")


def plot_class_distribution_pie(class_counts, title, out_path):
    labels = list(class_counts.keys())
    vals = np.asarray(list(class_counts.values()), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _ = ax.pie(vals, startangle=90, wedgeprops={"width": 0.35})
    ax.set_title(title)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    _save(fig, Path(out_path), "class distribution pie")


def plot_fold_summary(fold_metrics_list, model_name, out_dir):
    keys = [
        ("accuracy_pct", "Accuracy"),
        ("f1_macro", "F1"),
        ("balanced_accuracy", "Balanced Acc"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("specificity_macro", "Specificity"),
    ]

    folds = np.arange(1, len(fold_metrics_list) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (k, title) in zip(axes, keys):
        vals = np.asarray([fm.get(k, 0.0) for fm in fold_metrics_list], dtype=float)
        mean = float(np.mean(vals)) if len(vals) else 0.0
        std = float(np.std(vals)) if len(vals) else 0.0

        ax.bar(folds, vals, color="tab:blue", alpha=0.8)
        ax.axhline(mean, color="black", linestyle="--", linewidth=1.0, label=f"mean={mean:.4f}")
        ax.fill_between(
            [0.5, len(folds) + 0.5],
            [mean - std, mean - std],
            [mean + std, mean + std],
            color="gray",
            alpha=0.2,
            label=f"±std={std:.4f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    out_path = Path(out_dir) / "models" / model_name / f"{model_name}_cv_summary.png"
    _save(fig, out_path, "fold summary")


def plot_model_comparison(results_dict, out_dir):
    model_names = list(results_dict.keys())
    metric_keys = ["accuracy_pct", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "specificity_macro", "roc_auc_macro"]

    x = np.arange(len(model_names))
    w = 0.1

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, k in enumerate(metric_keys):
        vals = [results_dict[m].get(k, 0.0) for m in model_names]
        ax.bar(x + (i - len(metric_keys) / 2) * w + w / 2, vals, width=w, label=k)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20)
    ax.set_title("Model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    out_path = Path(out_dir) / "plots" / "model_comparison.png"
    _save(fig, out_path, "model comparison")


def plot_gradcam_overlay(img_tensor, attn_map, pred_class, true_class, out_path):
    img = np.asarray(img_tensor)
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img / 255.0, 0, 1)

    heat = np.asarray(attn_map)
    if heat.ndim == 3:
        heat = heat.squeeze()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_rgb = plt.cm.jet(heat)[..., :3]

    if heat_rgb.shape[:2] != img.shape[:2]:
        heat_rgb = np.asarray(
            plt.imshow(heat, cmap="jet").get_array()
        )

    overlay = np.clip(0.6 * img + 0.4 * heat_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(heat, cmap="jet")
    axes[1].set_title("Attention Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay pred={pred_class} true={true_class}")
    for ax in axes:
        ax.axis("off")

    _save(fig, Path(out_path), "GradCAM overlay")


def plot_learning_rate_schedule(lr_history, out_path):
    lr_history = np.asarray(lr_history, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(1, len(lr_history) + 1), lr_history, color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log)")
    ax.set_title("Learning rate schedule")
    ax.grid(alpha=0.25)
    _save(fig, Path(out_path), "learning rate schedule")


def plot_class_accuracy(y_true, y_pred, model_name, fold, out_dir):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = _default_class_names(9)
    accs = []

    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(float((y_pred[mask] == y_true[mask]).mean() * 100.0))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(classes, accs, color=["tab:orange" if v < 90.0 else "tab:blue" for v in accs])
    ax.axhline(90.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Accuracy %")
    ax.set_title("Per-class accuracy")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_class_accuracy.png"
    _save(fig, out_path, "class accuracy")


def plot_precision_recall_curve(y_true, y_probs, model_name, fold, out_dir):
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    classes = _default_class_names(y_probs.shape[1])

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{cls} AUC={pr_auc:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    out_path = fold_dir / f"fold{fold}_precision_recall_curves.png"
    _save(fig, out_path, "precision-recall curves")


def plot_training_phase_comparison(history, out_dir):
    train_acc = _safe_get(history, "train_acc")
    val_acc = _safe_get(history, "val_acc")
    phase1_epochs = int(history.get("phase1_epochs", 5))
    epochs = np.arange(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, train_acc, label="train_acc")
    ax.plot(epochs, val_acc, label="val_acc")

    if len(epochs) > 0:
        ax.axvspan(1, min(phase1_epochs, len(epochs)), alpha=0.15, color="lightblue", label="Phase 1")
        ax.axvspan(min(phase1_epochs + 1, len(epochs)), len(epochs), alpha=0.10, color="lightgreen", label="Phase 2")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Training phases comparison")
    ax.legend()
    ax.grid(alpha=0.25)

    _save(fig, Path(out_dir), "training phase comparison")


def save_all_fold_plots(history, y_true, y_pred, y_probs, per_class, model_name, fold, out_dir):
    class_names = list(per_class.keys()) if per_class else _default_class_names()

    plot_training_history(history, model_name, fold, out_dir)
    plot_confusion_matrix(y_true, y_pred, model_name, fold, out_dir, class_names)
    plot_confusion_matrix_norm(y_true, y_pred, model_name, fold, out_dir, class_names)
    plot_roc_curves(y_true, y_probs, model_name, fold, out_dir, class_names)
    plot_per_class_metrics(per_class, model_name, fold, out_dir)

    # Plot 10 requires external GradCAM tensors; emit placeholder if not provided.
    fold_dir = _fold_plot_dir(model_name, fold, out_dir)
    dummy_img = np.zeros((300, 300, 3), dtype=np.float32)
    dummy_heat = np.zeros((300, 300), dtype=np.float32)
    plot_gradcam_overlay(
        dummy_img,
        dummy_heat,
        pred_class="N/A",
        true_class="N/A",
        out_path=fold_dir / f"fold{fold}_gradcam_overlay.png",
    )

    plot_class_accuracy(y_true, y_pred, model_name, fold, out_dir)
    plot_precision_recall_curve(y_true, y_probs, model_name, fold, out_dir)

    phase_out = fold_dir / f"fold{fold}_training_phases.png"
    plot_training_phase_comparison(history, phase_out)

    # Model-level LR schedule path.
    model_lr_out = Path(out_dir) / "models" / model_name / f"{model_name}_lr_schedule.png"
    plot_learning_rate_schedule(_safe_get(history, "lr"), model_lr_out)


def save_metrics_txt(metrics: dict[str, Any], txt_path: str | Path) -> None:
    path = Path(txt_path)
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved metrics text: {path}")
