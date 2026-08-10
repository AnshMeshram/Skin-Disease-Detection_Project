from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.dataset import CLASS_NAMES, build_dataframe, create_holdout_dataloaders
from src.evaluate import evaluate_model, save_results
from src.model_factory import build_model
from src.plots import save_all_fold_plots


def main() -> None:
    root = ROOT
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    model_name = str(config.get("model", "efficientnet_b3"))
    fold = 0

    eval_config = dict(config)
    eval_config["num_workers"] = 0
    eval_config["val_num_workers"] = 0
    eval_config["persistent_workers"] = False
    eval_config["prefetch_factor"] = 2
    eval_config["pin_memory"] = bool(torch.cuda.is_available())

    df = build_dataframe(config, prefer_preprocessed=bool(config.get("preprocessing", {}).get("use_preprocessed", True)))
    _, _, test_dl, _ = create_holdout_dataloaders(df, eval_config, preprocess_fn=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, num_classes=int(config.get("num_classes", 9)), pretrained=False).to(device)
    ckpt_path = out_root / model_name / f"fold_{fold}" / "best.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)

    metrics, y_true, y_pred, y_probs = evaluate_model(model, test_dl, CLASS_NAMES)

    history_path = out_root / model_name / f"fold_{fold}" / "history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    history["phase1_epochs"] = int(history.get("phase1_epochs", config.get("phase1_epochs", 0)))

    save_results(metrics, out_root / "models" / model_name / "holdout" / "metrics.json")
    with open(out_root / "models" / model_name / "holdout" / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    save_all_fold_plots(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        per_class=metrics.get("per_class", {}),
        model_name=model_name,
        fold=fold,
        out_dir=out_root,
    )

    print("DONE")
    print(f"accuracy_pct={metrics['accuracy_pct']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.6f}")


if __name__ == "__main__":
    main()