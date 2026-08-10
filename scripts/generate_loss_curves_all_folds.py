from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_series(history_path: Path, key: str) -> list[float]:
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    series = data.get(key, [])
    return [float(v) for v in series] if isinstance(series, list) else []


def main() -> None:
    root = Path("outputs") / "efficientnet_b3"
    fold_dirs = sorted(p for p in root.glob("fold_*") if p.is_dir())
    if not fold_dirs:
        raise SystemExit("No fold directories found under outputs/efficientnet_b3")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_train, ax_val = axes

    plotted = 0
    for fold_dir in fold_dirs:
        history_path = fold_dir / "history.json"
        train_loss = load_series(history_path, "train_loss")
        val_loss = load_series(history_path, "val_loss")
        if not train_loss and not val_loss:
            continue

        label = fold_dir.name.replace("fold_", "Fold ")
        if train_loss:
            ax_train.plot(train_loss, label=label)
        if val_loss:
            ax_val.plot(val_loss, label=label)
        plotted += 1

    if plotted == 0:
        raise SystemExit("No loss curves found in fold history.json files")

    ax_train.set_title("EfficientNet-B3 Training Loss (All Folds)")
    ax_train.set_ylabel("Loss")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(ncols=3, fontsize=8, frameon=False)

    ax_val.set_title("EfficientNet-B3 Validation Loss (All Folds)")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(ncols=3, fontsize=8, frameon=False)

    out_dir = Path("outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss curves all folds.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
