import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_analysis import run_full_analysis
from src.data_verify import main as verify_data_main
from src.dataset import CLASS_NAMES, build_dataframe, create_fold_dataloaders, create_holdout_dataloaders
from src.evaluate import evaluate_model, save_results
from src.ensemble import run_ensemble_stage
from src.feature_extraction import run_feature_analysis
from src.gradcam import generate_gradcam_grid
from src.live_detection import LiveDetector, load_runtime_predictor
from src.losses import verify_loss
from src.model_factory import build_model
from src.plots import plot_fold_summary, plot_model_comparison, save_all_fold_plots, save_metrics_txt
from src.preprocessing import run_full_pipeline
from src.segmentation_classical import analyse_dataset_segmentation
from src.trainer import train_all_folds
from src.trainer import Trainer
from src.unet import UNet, compare_segmentation_methods, train_unet
from src.utils import check_environment


PREDICTION_LABELS = {
    "MEL": "melanoma",
    "NV": "nevus",
    "BCC": "basal_cell_carcinoma",
    "AK": "actinic_keratosis",
    "BKL": "benign_keratosis",
    "DF": "dermatofibroma",
    "VASC": "vascular_lesion",
    "SCC": "squamous_cell_carcinoma",
    "Healthy": "healthy",
}


def fix_seeds(seed=42):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_csv_columns(config: dict) -> None:
    df = pd.read_csv(config["data"]["isic_ground_truth"], nrows=0)
    cols = set(df.columns.tolist())
    required_a = {"MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"}
    required_b = {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"}
    if not (required_a.issubset(cols) or required_b.issubset(cols)):
        raise RuntimeError(f"Invalid CSV columns: {sorted(cols)}")


def verify_class_distribution(config: dict) -> None:
    df = build_dataframe(config, prefer_preprocessed=False)
    counts = df["label"].value_counts().sort_index()
    missing = [CLASS_NAMES[i] for i in range(9) if counts.get(i, 0) <= 0]
    if missing:
        raise RuntimeError(f"Missing classes: {missing}")


def verify_sampler_no_nan(config: dict) -> None:
    df = build_dataframe(config, prefer_preprocessed=False)
    counts = df["label"].value_counts().sort_index()
    w = 1.0 / counts.reindex(range(9), fill_value=1).to_numpy(dtype=np.float32)
    if np.isnan(w).any() or np.isinf(w).any():
        raise RuntimeError("Sampler weights contain NaN/Inf")


def verify_loss_function(config: dict) -> None:
    weights = [1.0] * 9
    ok = verify_loss(config, weights, device="cpu")
    if not ok:
        raise RuntimeError("Loss is not finite")


def verify_model_shapes(config: dict) -> None:
    model_name = str(config.get("model", "efficientnet_b3"))
    model = build_model(model_name, num_classes=9, pretrained=False)
    x = torch.randn(2, 3, 300, 300)
    y = model(x)
    if y.shape != (2, 9):
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")


def verify_preprocessing(config: dict) -> None:
    img_path = Path(config["data"]["isic_images_dir"]) / "ISIC_0000000.jpg"
    if not img_path.exists():
        sample = next(Path(config["data"]["isic_images_dir"]).glob("*.jpg"), None)
        if sample is None:
            raise RuntimeError("No sample ISIC image found")
        img_path = sample
    out = run_full_pipeline(str(img_path))
    if out["step6_tensor"].shape[-2:] != (300, 300):
        raise RuntimeError("Preprocessing tensor shape mismatch")


def verify_unet_optional(config: dict) -> None:
    ckpt = Path(config.get("preprocessing", {}).get("unet_checkpoint", "outputs/unet/best_unet.pth"))
    if ckpt.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UNet().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)


def verify_all(config: dict) -> None:
    checks = [
        ("Python + CUDA", lambda c: check_environment()),
        ("Dataset CSV columns", verify_csv_columns),
        ("All 9 classes present", verify_class_distribution),
        ("Sampler weights", verify_sampler_no_nan),
        ("Loss not NaN", verify_loss_function),
        ("Model forward pass", verify_model_shapes),
        ("Preprocessing pipeline", verify_preprocessing),
        ("U-Net checkpoint", verify_unet_optional),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn(config)
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"\n  {passed}/{len(checks)} checks passed")
    if passed < len(checks):
        print("  Fix all ✗ checks before training.")
        sys.exit(1)
    print("  All checks PASSED — safe to train.\n")


def stage_preprocess(config: dict, workers: int = 0) -> None:
    df = build_dataframe(config, prefer_preprocessed=False)
    out_dir = Path(config["data"]["output_dir"]) / "preprocessed"
    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(df)
    skipped = 0
    processed = 0
    for idx, (_, r) in enumerate(df.iterrows(), start=1):
        p = Path(r["image_path"])
        proc_path = img_out / f"{r['image']}.png"
        mask_path = mask_out / f"{r['image']}.png"

        # Resume support: keep existing outputs and continue with missing files.
        if proc_path.exists() and mask_path.exists():
            rows.append({"image": r["image"], "processed_path": str(proc_path), "mask_path": str(mask_path)})
            skipped += 1
            if idx % 250 == 0 or idx == total:
                print(f"Preprocess progress: {idx}/{total} | processed={processed} skipped={skipped}")
            continue

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        res = run_full_pipeline(img)
        proc = res["step5_no_hair"]
        cv2.imwrite(str(proc_path), proc)

        cv2.imwrite(str(mask_path), np.zeros((300, 300), dtype=np.uint8))

        rows.append({"image": r["image"], "processed_path": str(proc_path), "mask_path": str(mask_path)})
        processed += 1
        if idx % 250 == 0 or idx == total:
            print(f"Preprocess progress: {idx}/{total} | processed={processed} skipped={skipped}")

    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"Saved manifest: {out_dir / 'manifest.csv'}")


def stage_train(config: dict, resume: bool = False, fold: int | None = None, warm_start: str = "") -> None:
    df = build_dataframe(config, prefer_preprocessed=bool(config.get("preprocessing", {}).get("use_preprocessed", True)))
    fold_ids = [fold] if fold is not None else None
    summary = train_all_folds(config, df, preprocess_fn=None, resume=resume, fold_ids=fold_ids, warm_start=warm_start)
    out = Path("outputs") / "training_summary.json"
    print(f"Training complete. Summary entries: {len(summary)}")
    print(f"Summary path: {out}")


def _resolve_models(config: dict, model_override: str | None = None) -> list[str]:
    if model_override and model_override.lower() != "all":
        return [model_override]
    models = config.get("architectures") or [config.get("model", "efficientnet_b3")]
    if isinstance(models, str):
        return [models]
    return list(models)


def _split_strategy(config: dict) -> str:
    split_cfg = config.get("split", {})
    strategy = config.get("split_strategy", split_cfg.get("strategy", "holdout"))
    return str(strategy).strip().lower()


def _load_fold_history(out_root: Path, model_name: str, fold_idx: int) -> dict:
    history_path = out_root / model_name / f"fold_{fold_idx}" / "history.json"
    if not history_path.exists():
        return {"phase1_epochs": 0}
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, dict):
        return {"phase1_epochs": 0}
    history["phase1_epochs"] = int(history.get("phase1_epochs", 0))
    return history


def stage_evaluate(config: dict, model_override: str = "all", fold: int | None = None) -> None:
    models = _resolve_models(config, model_override)
    strategy = _split_strategy(config)
    folds = int(config.get("folds", 5))
    fold_ids = [fold] if fold is not None else list(range(folds))

    out_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    prefer_pre = bool(config.get("preprocessing", {}).get("use_preprocessed", True))
    df = build_dataframe(config, prefer_preprocessed=prefer_pre)
    eval_config = dict(config)
    eval_config["num_workers"] = 0
    eval_config["val_num_workers"] = 0
    eval_config["persistent_workers"] = False
    eval_config["prefetch_factor"] = 2
    eval_config["pin_memory"] = bool(torch.cuda.is_available())

    model_macro = {}
    for model_name in models:
        if strategy == "holdout":
            print(f"Evaluating {model_name} | holdout test split")
            _tr_dl, _val_dl, test_dl, _cw = create_holdout_dataloaders(df, eval_config, preprocess_fn=None)

            model = build_model(model_name, num_classes=int(config.get("num_classes", 9)), pretrained=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            ckpt_path = Path("outputs") / model_name / "fold_0" / "best.pth"
            if not ckpt_path.exists():
                print(f"Skipping holdout eval: checkpoint missing at {ckpt_path}")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)

            metrics, y_true, y_pred, y_probs = evaluate_model(model, test_dl, CLASS_NAMES)

            out_dir = out_root / "models" / model_name / "holdout"
            json_out = out_dir / "metrics.json"
            txt_out = out_dir / "metrics.txt"
            save_results(metrics, json_out)
            save_metrics_txt(metrics, txt_out)
            save_all_fold_plots(
                history=_load_fold_history(out_root, model_name, 0),
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                per_class=metrics.get("per_class", {}),
                model_name=model_name,
                fold=0,
                out_dir=out_root,
            )

            model_macro[model_name] = {
                "accuracy_pct": float(metrics["accuracy_pct"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "specificity_macro": float(metrics["specificity_macro"]),
                "roc_auc_macro": float(metrics["roc_auc_macro"]),
            }
            continue

        fold_metrics = []
        for fidx in fold_ids:
            print(f"Evaluating {model_name} | fold {fidx}")
            _tr_dl, val_dl, _cw = create_fold_dataloaders(df, eval_config, fold_idx=fidx, preprocess_fn=None)

            model = build_model(model_name, num_classes=int(config.get("num_classes", 9)), pretrained=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            ckpt_path = Path("outputs") / model_name / f"fold_{fidx}" / "best.pth"
            if not ckpt_path.exists():
                print(f"Skipping fold {fidx}: checkpoint missing at {ckpt_path}")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)

            metrics, y_true, y_pred, y_probs = evaluate_model(model, val_dl, CLASS_NAMES)
            fold_metrics.append(metrics)

            json_out = out_root / "models" / model_name / f"fold_{fidx}" / "metrics.json"
            txt_out = out_root / "models" / model_name / f"fold_{fidx}" / "metrics.txt"
            save_results(metrics, json_out)
            save_metrics_txt(metrics, txt_out)
            save_all_fold_plots(
                history={"phase1_epochs": int(config.get("phase1_epochs", 5))},
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                per_class=metrics.get("per_class", {}),
                model_name=model_name,
                fold=fidx,
                out_dir=out_root,
            )

        if fold_metrics:
            model_avg = {
                "accuracy_pct": float(np.mean([m["accuracy_pct"] for m in fold_metrics])),
                "balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
                "f1_macro": float(np.mean([m["f1_macro"] for m in fold_metrics])),
                "precision_macro": float(np.mean([m["precision_macro"] for m in fold_metrics])),
                "recall_macro": float(np.mean([m["recall_macro"] for m in fold_metrics])),
                "specificity_macro": float(np.mean([m["specificity_macro"] for m in fold_metrics])),
                "roc_auc_macro": float(np.mean([m["roc_auc_macro"] for m in fold_metrics])),
            }
            model_macro[model_name] = model_avg
            plot_fold_summary(fold_metrics, model_name, out_root)

            summary_path = out_root / "models" / model_name / "evaluation_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as fsum:
                json.dump({"model": model_name, "folds": fold_metrics, "mean": model_avg}, fsum, indent=2)

    if model_macro:
        plot_model_comparison(model_macro, out_root)
        model_cmp_path = out_root / "plots" / "model_comparison_metrics.json"
        model_cmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_cmp_path, "w", encoding="utf-8") as fcmp:
            json.dump(model_macro, fcmp, indent=2)
        print(f"Saved model comparison summary: {model_cmp_path}")


def stage_gradcam(config: dict, model_name: str, fold: int = 0, n_per_class: int = 3) -> None:
    prefer_pre = bool(config.get("preprocessing", {}).get("use_preprocessed", True))
    df = build_dataframe(config, prefer_preprocessed=prefer_pre)
    _tr_dl, val_dl, _cw = create_fold_dataloaders(df, config, fold_idx=fold, preprocess_fn=None)

    model = build_model(model_name, num_classes=int(config.get("num_classes", 9)), pretrained=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ckpt_path = Path("outputs") / model_name / f"fold_{fold}" / "best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    out = generate_gradcam_grid(
        model=model,
        dataloader=val_dl,
        class_names=CLASS_NAMES,
        out_dir=str(config.get("data", {}).get("output_dir", "outputs")),
        n_per_class=n_per_class,
        model_name=model_name,
    )
    print(f"GradCAM grid: {out}")


def stage_status(config: dict) -> None:
    print("Status")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Split strategy: {_split_strategy(config)}")
    print(f"Preprocessed manifest exists: {Path(config['data']['output_dir']) / 'preprocessed/manifest.csv'}")


def stage_predict(config: dict, image_path: str, model_name: str, use_ensemble: bool, fold: int, output_path: str = "") -> dict:
    predictor = load_runtime_predictor(
        config,
        model_name=model_name,
        use_ensemble=use_ensemble,
        fold=fold,
    )
    detector = LiveDetector(predictor, config)
    result = detector.predict_image(image_path)

    class_name = str(result["class"])
    payload = {
        "status": "success",
        "image": Path(image_path).name,
        "prediction": PREDICTION_LABELS.get(class_name, class_name.lower()),
        "confidence": float(result["confidence"]),
        "probabilities": {k: float(v) for k, v in result["probabilities"].items()},
        "class_index": int(CLASS_NAMES.index(class_name)) if class_name in CLASS_NAMES else -1,
        "is_uncertain": bool(result["is_uncertain"]),
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved prediction JSON: {out}")
    else:
        print(json.dumps(payload, indent=2))

    return payload


def stage_train_holdout(config: dict, resume: bool = False) -> None:
    models = config.get("architectures") or [config.get("model", "efficientnet_b3")]
    if isinstance(models, str):
        models = [models]
    prefer_pre = bool(config.get("preprocessing", {}).get("use_preprocessed", True))
    df = build_dataframe(config, prefer_preprocessed=prefer_pre)

    for model_name in models:
        print(f"\n=== Training {model_name} | stratified holdout 80/10/10 ===")
        tr_dl, val_dl, test_dl, class_weights = create_holdout_dataloaders(df, config, preprocess_fn=None)

        model = build_model(model_name, num_classes=int(config.get("num_classes", 9)), pretrained=bool(config.get("pretrained", True)))
        trainer = Trainer(model=model, config=config, class_weights=class_weights, fold=0)
        trainer.train(tr_dl, val_dl, model_name=model_name, resume=resume)

        best_ckpt = Path("outputs") / model_name / "fold_0" / "best.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=trainer.device)
            trainer.model.load_state_dict(ckpt["model_state"])
            metrics, _y_true, _y_pred, _y_probs = evaluate_model(trainer.model, test_dl, CLASS_NAMES)
            out = Path(config.get("data", {}).get("output_dir", "outputs")) / "models" / model_name / "holdout_test_metrics.json"
            save_results(metrics, out)
            print(f"Saved holdout test metrics: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skin Disease Detection Training/Evaluation")
    parser.add_argument("--stage", required=True, help="Stage to run (train, evaluate, status, etc.)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--analysis_stage", default="all", help="Analysis stage (all, dataset, preprocessing, etc.)")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for preprocessing")
    parser.add_argument("--model", default="", help="Model architecture override")
    parser.add_argument("--fold", type=int, default=-1, help="Fold index for k-fold or evaluation")
    parser.add_argument("--n_per_class", type=int, default=3, help="Number of GradCAMs per class")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID for live detection")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble for prediction")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest/best checkpoint")
    parser.add_argument("--image", default="", help="Image path for prediction")
    parser.add_argument("--output", default="", help="Output path for prediction JSON")
    parser.add_argument("--split_strategy", default="", help="Split strategy override (holdout, kfold)")
    parser.add_argument("--warm_start", default="", help="Path to checkpoint to warm start fold training from")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.split_strategy:
        config["split_strategy"] = args.split_strategy
    fix_seeds(int(config.get("seed", 42)))

    if args.stage == "verify":
        verify_all(config)
    elif args.stage == "analyse":
        run_full_analysis(config, stage=args.analysis_stage)
    elif args.stage == "preprocess":
        stage_preprocess(config, workers=args.workers)
    elif args.stage == "train_unet":
        train_unet(config, pseudo_mask_dir="outputs/preprocessed/masks", out_dir="outputs/unet")
    elif args.stage == "train":
        config = dict(config)
        config["architectures"] = _resolve_models(config, args.model)
        if args.warm_start:
            config["resume_from_best"] = True
        if _split_strategy(config) == "holdout":
            stage_train_holdout(config, resume=args.resume)
        else:
            fold = args.fold if args.fold >= 0 else None
            stage_train(config, resume=args.resume, fold=fold, warm_start=args.warm_start)
    elif args.stage == "status":
        stage_status(config)
    elif args.stage == "evaluate":
        fold = args.fold if args.fold >= 0 else None
        stage_evaluate(config, model_override=args.model, fold=fold)
    elif args.stage == "features":
        run_feature_analysis(config)
    elif args.stage == "gradcam":
        stage_gradcam(config, model_name=args.model, fold=max(args.fold, 0), n_per_class=max(1, args.n_per_class))
    elif args.stage == "ensemble":
        fold = max(args.fold, 0)
        if args.model and args.model.lower() != "all":
            archs = [x.strip() for x in args.model.split(",") if x.strip()]
        else:
            archs = None
        try:
            run_ensemble_stage(config, architectures=archs, fold_idx=fold)
        except FileNotFoundError as e:
            print(f"Ensemble cannot start: {e}")
            print("Train the requested architectures first, then rerun Stage 11.")
            sys.exit(1)
    elif args.stage == "predict":
        if not args.image:
            raise ValueError("--image is required for predict stage")
        try:
            stage_predict(
                config=config,
                image_path=args.image,
                model_name=args.model or config.get("model", "efficientnet_b3"),
                use_ensemble=args.use_ensemble,
                fold=max(args.fold, 0),
                output_path=args.output,
            )
        except Exception as e:
            print(f"Prediction failed: {e}")
            sys.exit(1)
    else:
        print("Unknown stage. Use --help for options.")
        sys.exit(1)
    help_text = "\n".join([
        "Commands summary:",
        "  python main.py --stage train    --config config.yaml --model efficientnet_b3",
        "  python main.py --stage train    --config config.yaml --model efficientnet_b3 --resume",
        "  python main.py --stage status   --config config.yaml --model efficientnet_b3",
        "  python main.py --stage verify   --config config.yaml",
        "  python main.py --stage evaluate --config config.yaml --model efficientnet_b3",
        "  python main.py --stage predict  --config config.yaml --image path/to/img.jpg",
        "  python main.py --stage live     --config config.yaml --model efficientnet_b3",
        "  python main.py --stage ensemble --config config.yaml",
        "  uvicorn src.api:app --host 0.0.0.0 --port 8080",
    ])

    parser = argparse.ArgumentParser(
        description="Skin disease detection pipeline",
        epilog=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--analysis_stage", default="all")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model", default="efficientnet_b3")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--n_per_class", type=int, default=3)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--split_strategy", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.split_strategy:
        config["split_strategy"] = args.split_strategy
    fix_seeds(int(config.get("seed", 42)))

    if args.stage == "verify":
        verify_all(config)
    elif args.stage == "analyse":
        run_full_analysis(config, stage=args.analysis_stage)
    elif args.stage == "preprocess":
        stage_preprocess(config, workers=args.workers)
    elif args.stage == "train_unet":
        train_unet(config, pseudo_mask_dir="outputs/preprocessed/masks", out_dir="outputs/unet")
    elif args.stage == "train":
        config = dict(config)
        config["architectures"] = _resolve_models(config, args.model)
        if args.warm_start:
            config["resume_from_best"] = True
        if _split_strategy(config) == "holdout":
            stage_train_holdout(config, resume=args.resume)
        else:
            fold = args.fold if args.fold >= 0 else None
            stage_train(config, resume=args.resume, fold=fold, warm_start=args.warm_start)
    elif args.stage == "status":
        stage_status(config)
    elif args.stage == "evaluate":
        fold = args.fold if args.fold >= 0 else None
        stage_evaluate(config, model_override=args.model, fold=fold)
    elif args.stage == "features":
        run_feature_analysis(config)
    elif args.stage == "gradcam":
        stage_gradcam(config, model_name=args.model, fold=max(args.fold, 0), n_per_class=max(1, args.n_per_class))
    elif args.stage == "ensemble":
        fold = max(args.fold, 0)
        if args.model and args.model.lower() != "all":
            archs = [x.strip() for x in args.model.split(",") if x.strip()]
        else:
            archs = None
        try:
            run_ensemble_stage(config, architectures=archs, fold_idx=fold)
        except FileNotFoundError as e:
            print(f"Ensemble cannot start: {e}")
            print("Train the requested architectures first, then rerun Stage 11.")
            sys.exit(1)
    elif args.stage == "predict":
        if not args.image:
            raise ValueError("--image is required for predict stage")
        try:
            stage_predict(
                config=config,
                image_path=args.image,
                model_name=args.model,
                use_ensemble=bool(args.use_ensemble),
                fold=max(args.fold, 0),
                output_path=args.output,
            )
        except FileNotFoundError as e:
            print(f"Predict stage cannot start: {e}")
            print("Train requested model(s) first and rerun predict stage.")
            sys.exit(1)
    elif args.stage == "live":
        fold = max(args.fold, 0)
        try:
            predictor = load_runtime_predictor(
                config,
                model_name=args.model,
                use_ensemble=bool(args.use_ensemble),
                fold=fold,
            )
        except FileNotFoundError as e:
            print(f"Live stage cannot start: {e}")
            print("Train requested model(s) first and rerun live stage.")
            sys.exit(1)

        detector = LiveDetector(predictor, config)
        detector.run_webcam(camera_id=int(args.camera_id))
    elif args.stage == "api":
        print("Run the API with: uvicorn src.api:app --host 0.0.0.0 --port 8080")
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
