from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.dataset import CLASS_NAMES, get_albumentations_val
from src.ensemble import EnsembleModel, load_best_checkpoints
from src.gradcam import GradCAM, get_target_layer
from src.model_factory import build_model
from src.preprocessing import preprocess_image


class LiveDetector:
    def __init__(self, model_or_ensemble, config, preprocess_fn=None):
        self.config = config
        self.model_or_ensemble = model_or_ensemble
        self.preprocess_fn = preprocess_fn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_enabled = bool(config.get("amp", True)) and self.device == "cuda"

        live_cfg = config.get("live", {})
        self.conf_threshold = float(live_cfg.get("confidence_threshold", 0.7))
        self.use_hair = bool(live_cfg.get("hair_removal", True))
        self.use_clahe = bool(live_cfg.get("clahe", True))
        self.gradcam_alpha = float(live_cfg.get("gradcam_alpha", 0.35))
        self.gradcam_enabled = bool(live_cfg.get("gradcam", True))

        self.class_names = list(CLASS_NAMES)
        self.healthy_name = "Healthy"

        self.transform = get_albumentations_val(config)

        self.is_ensemble = hasattr(model_or_ensemble, "predict") and not isinstance(model_or_ensemble, torch.nn.Module)

        if self.is_ensemble:
            self.ensemble = model_or_ensemble
            self.model = None
        else:
            self.model = model_or_ensemble.to(self.device)
            self.model.eval()
            self.ensemble = None

        self.gradcam = None
        if self.gradcam_enabled and self.model is not None:
            try:
                model_name = str(config.get("model", "efficientnet_b3"))
                layer_name = get_target_layer(model_name, self.model)
                self.gradcam = GradCAM(self.model, layer_name)
            except Exception:
                # Keep live mode running even if Grad-CAM hook setup fails.
                self.gradcam = None

    def _prepare_input(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame")

        if self.preprocess_fn is not None:
            proc_bgr = self.preprocess_fn(frame_bgr)
        else:
            proc_bgr = preprocess_image(frame_bgr, apply_hair=self.use_hair, apply_clahe=self.use_clahe)

        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(image=rgb)["image"].unsqueeze(0)
        return x.to(self.device, non_blocking=True)

    def _infer_probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with autocast(enabled=self.amp_enabled):
                if self.is_ensemble:
                    probs = self.ensemble.predict(x)
                else:
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=1)
        return probs

    def predict_frame(self, frame_bgr: np.ndarray) -> dict:
        x = self._prepare_input(frame_bgr)
        probs = self._infer_probs(x)

        conf, cls_idx = probs.max(dim=1)
        conf_v = float(conf.item())
        idx_v = int(cls_idx.item())
        cls_name = self.class_names[idx_v]

        prob_np = probs[0].detach().cpu().numpy().astype(np.float64)
        prob_dict = {name: float(prob_np[i]) for i, name in enumerate(self.class_names)}

        is_uncertain = conf_v < self.conf_threshold
        if is_uncertain:
            color = "yellow"
        elif cls_name == self.healthy_name:
            color = "green"
        else:
            color = "red"

        gradcam_overlay_bgr = None
        if self.gradcam is not None:
            try:
                heat = self.gradcam.generate(x, class_idx=idx_v)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ov_rgb = self.gradcam.overlay(rgb, heat, alpha=self.gradcam_alpha)
                gradcam_overlay_bgr = cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                gradcam_overlay_bgr = None

        return {
            "class": cls_name,
            "confidence": conf_v,
            "probabilities": prob_dict,
            "is_uncertain": is_uncertain,
            "color": color,
            "gradcam_overlay_bgr": gradcam_overlay_bgr,
        }

    def _to_bgr(self, color_name: str) -> tuple[int, int, int]:
        if color_name == "green":
            return (50, 200, 50)
        if color_name == "yellow":
            return (0, 220, 255)
        return (0, 70, 255)

    def _draw_prob_bars(self, frame: np.ndarray, prob_dict: dict[str, float]) -> None:
        h, w = frame.shape[:2]
        panel_w = min(240, max(180, int(w * 0.28)))
        panel_h = min(220, max(160, int(h * 0.45)))
        x1 = w - panel_w - 12
        y1 = 12
        x2 = w - 12
        y2 = y1 + panel_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), thickness=-1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)
        cv2.putText(frame, "Probabilities", (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        classes = list(prob_dict.keys())
        n = len(classes)
        avail_h = panel_h - 30
        row_h = max(12, avail_h // n)
        bar_max_w = panel_w - 75

        for i, cls in enumerate(classes):
            p = float(prob_dict[cls])
            y = y1 + 28 + i * row_h
            bw = int(bar_max_w * p)

            cv2.putText(frame, cls[:6], (x1 + 6, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1 + 58, y), (x1 + 58 + bar_max_w, y + 8), (50, 50, 50), thickness=-1)
            cv2.rectangle(frame, (x1 + 58, y), (x1 + 58 + bw, y + 8), (80, 180, 255), thickness=-1)

    def draw_overlay(self, frame_bgr, result: dict) -> np.ndarray:
        canvas = frame_bgr.copy()
        if result.get("gradcam_overlay_bgr") is not None:
            canvas = result["gradcam_overlay_bgr"].copy()

        color_bgr = self._to_bgr(result["color"])
        h, w = canvas.shape[:2]

        cv2.rectangle(canvas, (8, 8), (w - 8, h - 8), color_bgr, thickness=3)

        label = f"{result['class']}: {result['confidence'] * 100.0:.1f}%"
        cv2.rectangle(canvas, (15, 12), (15 + 360, 48), (10, 10, 10), thickness=-1)
        cv2.putText(canvas, label, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_bgr, 2, cv2.LINE_AA)

        if result["is_uncertain"]:
            cv2.rectangle(canvas, (15, 52), (245, 82), (0, 0, 0), thickness=-1)
            cv2.putText(canvas, "UNCERTAIN", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2, cv2.LINE_AA)

        self._draw_prob_bars(canvas, result["probabilities"])
        return canvas

    def run_webcam(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        win = "Skin Disease Detection"
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = self.predict_frame(frame)
            out = self.draw_overlay(frame, result)
            cv2.imshow(win, out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def predict_image(self, image_path: str) -> dict:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        return self.predict_frame(frame)


def _load_single_model(config: dict, model_name: str, fold: int = 0):
    num_classes = int(config.get("num_classes", 9))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)

    out_root = Path(config.get("data", {}).get("output_dir", "outputs"))
    candidates = [
        out_root / model_name / f"fold_{fold}" / "best.pth",
        out_root / model_name / f"fold_{fold}" / "latest.pth",
        out_root / "models" / model_name / f"fold_{fold}" / "best.pth",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found for {model_name} (fold {fold})")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Loaded model {model_name} from {ckpt_path}")
    return model


def load_runtime_predictor(config: dict, model_name: str = "efficientnet_b3", use_ensemble: bool = False, fold: int = 0):
    if use_ensemble:
        archs = config.get("architectures")
        if not archs:
            archs = ["efficientnet_b3", "inception_v3", "convnext_tiny"]
        if isinstance(archs, str):
            archs = [archs]

        loaded = load_best_checkpoints(config, archs)
        models = [m for m, _ in loaded]

        weights = [1.0 / len(models)] * len(models)
        temperatures = [1.0] * len(models)

        ens_metrics_path = Path(config.get("data", {}).get("output_dir", "outputs")) / "ensemble" / "ensemble_metrics.json"
        if ens_metrics_path.exists():
            try:
                import json

                with open(ens_metrics_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                p_archs = payload.get("architectures", [])
                if [a.lower() for a in p_archs] == [a.lower() for a in archs]:
                    weights = payload.get("weights", weights)
                    temperatures = payload.get("temperatures", temperatures)
                    print(f"Using saved ensemble calibration from {ens_metrics_path}")
            except Exception:
                pass

        return EnsembleModel(models=models, weights=weights, temperatures=temperatures)

    return _load_single_model(config, model_name=model_name, fold=fold)
