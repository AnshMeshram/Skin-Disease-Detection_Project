from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.dataset import build_dataframe, create_fold_dataloaders, CLASS_NAMES
from src.model_factory import build_model


def _load_checkpoint(model, ckpt_path: Path, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def _extract_features(model, loader, device: str, max_samples: int = 1200):
    features = []
    labels = []
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            x = model.backbone(imgs)
            x = model.sa(x)
            x = model.pool(x).flatten(1)
            feats = x.detach().cpu().numpy()
            labs = lbls.detach().cpu().numpy()

            if total + len(feats) > max_samples:
                keep = max_samples - total
                feats = feats[:keep]
                labs = labs[:keep]

            features.append(feats)
            labels.append(labs)
            total += len(feats)

            if total >= max_samples:
                break

    if not features:
        return np.empty((0,)), np.empty((0,))

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    config_path = Path("config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = "efficientnet_b3"
    fold_idx = 4

    df = build_dataframe(config, prefer_preprocessed=bool(config.get("preprocessing", {}).get("use_preprocessed", True)))
    _train_dl, val_dl, _cw = create_fold_dataloaders(df, config, fold_idx=fold_idx, preprocess_fn=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, num_classes=9, pretrained=False).to(device)

    ckpt_path = Path("outputs") / model_name / f"fold_{fold_idx}" / "best.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    X, y = _extract_features(model, val_dl, device=device, max_samples=1200)
    if X.size == 0:
        raise SystemExit("No features extracted from validation set")

    perplexity = max(5, min(30, X.shape[0] // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)

    colors = [
        "#e11d48",  # MEL
        "#2563eb",  # NV
        "#d97706",  # BCC
        "#7c3aed",  # AK
        "#059669",  # BKL
        "#db2777",  # DF
        "#4b5563",  # VASC
        "#ea580c",  # SCC
        "#10b981",  # Healthy
    ]

    plt.figure(figsize=(12, 10))
    for i, name in enumerate(CLASS_NAMES):
        mask = y == i
        if not np.any(mask):
            continue
        plt.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            label=name,
            alpha=0.75,
            s=40,
            edgecolors="white",
            linewidths=0.4,
            color=colors[i],
        )

    plt.title("t-SNE: EfficientNet-B3 Penultimate Features (Fold 4 Val)", fontsize=14, fontweight="bold", pad=18)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    out_dir = Path("outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tsne embeddings.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
