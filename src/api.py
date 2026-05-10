from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.dataset import CLASS_NAMES
from src.live_detection import LiveDetector, load_runtime_predictor


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

app = FastAPI(title="Skin Disease Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_detector: LiveDetector | None = None
_config = None
_class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]
_model_loading = False
_model_error: str | None = None


def _checkpoint_candidates(model_name: str, fold: int) -> list[Path]:
    out_root = Path(_config.get("data", {}).get("output_dir", "outputs")) if _config else Path("outputs")
    return [
        out_root / model_name / f"fold_{fold}" / "best.pth",
        out_root / model_name / f"fold_{fold}" / "latest.pth",
        out_root / "models" / model_name / f"fold_{fold}" / "best.pth",
    ]


def _ensure_checkpoint_available(model_name: str, fold: int) -> Path | None:
    candidates = _checkpoint_candidates(model_name, fold)
    existing = next((p for p in candidates if p.exists()), None)
    if existing is not None:
        return existing

    weights_url = os.environ.get("MODEL_WEIGHTS_URL")
    if not weights_url:
        return None

    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(weights_url, target)
        return target if target.exists() else None
    except Exception as exc:
        print(f"Weights download failed: {exc}")
        return None


def _load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_monitor_run(model: str | None, fold: int | None) -> tuple[dict, str, int, Path]:
    cfg = _config if _config is not None else _load_config("config.yaml")
    model_name = str(model or cfg.get("model", "efficientnet_b3"))
    monitor_cfg = cfg.get("monitor", {})
    fold_id = int(fold) if fold is not None else int(monitor_cfg.get("fold", 0))
    out_root = Path(cfg.get("data", {}).get("output_dir", "outputs"))
    run_dir = out_root / model_name / f"fold_{fold_id}"
    return cfg, model_name, fold_id, run_dir


def _read_history(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return {}
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_last(history: dict, key: str) -> float | None:
    series = history.get(key, [])
    if not isinstance(series, list) or not series:
        return None
    try:
        return float(series[-1])
    except Exception:
        return None


def _read_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        return {"path": str(path), "error": str(e)}

    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    return {
        "path": str(path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "phase": ckpt.get("phase") if isinstance(ckpt, dict) else None,
        "best_bal_acc": ckpt.get("best_bal_acc") if isinstance(ckpt, dict) else None,
        "metrics": metrics,
    }


def _tail(series: list, max_len: int = 200) -> list:
    if not isinstance(series, list):
        return []
    if len(series) <= max_len:
        return series
    return series[-max_len:]


def _load_model_sync():
    global _model, _config, _detector, _model_loading, _model_error
    _model_loading = True
    _model_error = None
    try:
        _config = _load_config("config.yaml")
        model_name = str(_config.get("model", "efficientnet_b3"))
        live_cfg = _config.get("live", {})
        use_ensemble = bool(live_cfg.get("use_ensemble", False))
        fold = int(live_cfg.get("fold", 0))

        if use_ensemble:
            archs = _config.get("architectures") or ["efficientnet_b3", "inception_v3", "convnext_tiny"]
            if isinstance(archs, str):
                archs = [archs]
            for arch in archs:
                _ensure_checkpoint_available(str(arch), fold)
        else:
            _ensure_checkpoint_available(model_name, fold)

        _model = load_runtime_predictor(
            _config,
            model_name=model_name,
            use_ensemble=use_ensemble,
            fold=fold,
        )
        _detector = LiveDetector(_model, _config)
        print(f"Model loaded: {model_name} fold {fold}")
    except Exception as e:
        _model = None
        _detector = None
        _model_error = str(e)
        print(f"API model loading warning: model not loaded ({e})")
    finally:
        _model_loading = False


@app.on_event("startup")
async def load_model():
    asyncio.create_task(asyncio.to_thread(_load_model_sync))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_loading": _model_loading,
        "model_error": _model_error,
        "classes": 9,
    }


@app.get("/metrics/latest")
async def latest_metrics(model: str | None = Query(None), fold: int | None = Query(None)):
    _, model_name, fold_id, run_dir = _resolve_monitor_run(model, fold)
    history = _read_history(run_dir)

    latest_ckpt = _read_checkpoint(run_dir / "latest.pth")
    best_ckpt = _read_checkpoint(run_dir / "best.pth")

    train_acc = _tail(history.get("train_acc", []))
    val_acc = _tail(history.get("val_acc", []))
    val_bal = _tail(history.get("val_balanced_accuracy", []))

    payload = {
        "model": model_name,
        "fold": fold_id,
        "run_dir": str(run_dir),
        "current": {
            "epoch": len(history.get("train_acc", [])),
            "train_acc": _safe_last(history, "train_acc"),
            "val_acc": _safe_last(history, "val_acc"),
            "val_balanced_accuracy": _safe_last(history, "val_balanced_accuracy"),
            "val_f1": _safe_last(history, "val_f1"),
            "lr": _safe_last(history, "lr"),
        },
        "history": {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_balanced_accuracy": val_bal,
        },
        "latest_checkpoint": latest_ckpt,
        "best_checkpoint": best_ckpt,
        "updated_at": f"{datetime.utcnow().isoformat()}Z",
    }
    return payload


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Live Training Accuracy</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #111827;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #34d399;
            --border: #1f2937;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            background: radial-gradient(circle at 10% 10%, #1f2937 0%, #0f172a 45%, #0b1020 100%);
            color: var(--text);
        }
        .container {
            max-width: 980px;
            margin: 40px auto;
            padding: 0 16px 40px;
        }
        h1 { margin: 0 0 6px; font-size: 28px; }
        .sub { color: var(--muted); margin-bottom: 18px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 6px 0;
            font-size: 14px;
        }
        .metric span:last-child { font-weight: 600; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            background: rgba(56,189,248,0.2);
            color: var(--accent);
            font-size: 12px;
            margin-left: 6px;
        }
        .chart {
            width: 100%;
            height: 220px;
            margin-top: 8px;
            background: #0b1224;
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        .legend {
            display: flex;
            gap: 12px;
            margin-top: 8px;
            font-size: 12px;
            color: var(--muted);
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .dot.train { background: var(--accent); }
        .dot.val { background: var(--accent-2); }
        .status { color: var(--muted); font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Training Accuracy</h1>
        <div class="sub" id="run-label">Waiting for data...</div>

        <div class="grid">
            <div class="card">
                <div><strong>Current</strong><span class="badge" id="epoch">epoch -</span></div>
                <div class="metric"><span>Train acc</span><span id="train-acc">-</span></div>
                <div class="metric"><span>Val acc</span><span id="val-acc">-</span></div>
                <div class="metric"><span>Val bal acc</span><span id="val-bal">-</span></div>
                <div class="metric"><span>Val F1</span><span id="val-f1">-</span></div>
                <div class="metric"><span>LR</span><span id="lr">-</span></div>
            </div>
            <div class="card">
                <div><strong>Best checkpoint</strong></div>
                <div class="metric"><span>Epoch</span><span id="best-epoch">-</span></div>
                <div class="metric"><span>Best bal acc</span><span id="best-bal">-</span></div>
                <div class="metric"><span>Val acc</span><span id="best-val-acc">-</span></div>
                <div class="metric"><span>Val F1</span><span id="best-val-f1">-</span></div>
                <div class="metric"><span>Phase</span><span id="best-phase">-</span></div>
            </div>
            <div class="card">
                <div><strong>Latest checkpoint</strong></div>
                <div class="metric"><span>Epoch</span><span id="latest-epoch">-</span></div>
                <div class="metric"><span>Val acc</span><span id="latest-val-acc">-</span></div>
                <div class="metric"><span>Val F1</span><span id="latest-val-f1">-</span></div>
                <div class="metric"><span>Phase</span><span id="latest-phase">-</span></div>
            </div>
        </div>

        <div class="card">
            <div><strong>Accuracy trend</strong></div>
            <svg id="acc-chart" class="chart" viewBox="0 0 600 220" preserveAspectRatio="none"></svg>
            <div class="legend">
                <span><span class="dot train"></span>Train acc</span>
                <span><span class="dot val"></span>Val acc</span>
            </div>
        </div>

        <div class="status" id="status">Last updated: -</div>
    </div>

    <script>
        const formatNum = (v, digits = 2, suffix = "") => {
            if (v === null || v === undefined || Number.isNaN(v)) return "-";
            return Number(v).toFixed(digits) + suffix;
        };

        const safe = (v) => (v === null || v === undefined ? "-" : v);

        function buildPoints(values, width, height) {
            if (!values || values.length < 2) return "";
            const min = Math.min(...values);
            const max = Math.max(...values);
            const span = (max - min) || 1;
            return values.map((v, i) => {
                const x = (i / (values.length - 1)) * width;
                const y = height - ((v - min) / span) * height;
                return `${x.toFixed(2)},${y.toFixed(2)}`;
            }).join(" ");
        }

        function renderChart(train, val) {
            const svg = document.getElementById("acc-chart");
            const width = 600;
            const height = 220;
            const trainPts = buildPoints(train, width, height);
            const valPts = buildPoints(val, width, height);
            const grid = [0, 25, 50, 75, 100].map((pct) => {
                const y = height - (pct / 100) * height;
                return `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="#1f2937" stroke-width="1" />`;
            }).join("");

            svg.innerHTML = `
                <rect width="${width}" height="${height}" fill="#0b1224" />
                ${grid}
                ${trainPts ? `<polyline points="${trainPts}" fill="none" stroke="#38bdf8" stroke-width="2" />` : ""}
                ${valPts ? `<polyline points="${valPts}" fill="none" stroke="#34d399" stroke-width="2" />` : ""}
            `;
        }

        async function refresh() {
            const qs = window.location.search || "";
            try {
                const res = await fetch(`/metrics/latest${qs}`);
                const data = await res.json();

                document.getElementById("run-label").textContent = `Model: ${data.model} | Fold: ${data.fold}`;
                document.getElementById("epoch").textContent = `epoch ${safe(data.current.epoch)}`;
                document.getElementById("train-acc").textContent = formatNum(data.current.train_acc, 2, "%");
                document.getElementById("val-acc").textContent = formatNum(data.current.val_acc, 2, "%");
                document.getElementById("val-bal").textContent = formatNum(data.current.val_balanced_accuracy, 4, "");
                document.getElementById("val-f1").textContent = formatNum(data.current.val_f1, 4, "");
                document.getElementById("lr").textContent = formatNum(data.current.lr, 6, "");

                const best = data.best_checkpoint || {};
                const bestMetrics = best.metrics || {};
                document.getElementById("best-epoch").textContent = safe(best.epoch);
                document.getElementById("best-bal").textContent = formatNum(best.best_bal_acc, 4, "");
                document.getElementById("best-val-acc").textContent = formatNum(bestMetrics.val_acc, 2, "%");
                document.getElementById("best-val-f1").textContent = formatNum(bestMetrics.val_f1, 4, "");
                document.getElementById("best-phase").textContent = safe(best.phase);

                const latest = data.latest_checkpoint || {};
                const latestMetrics = latest.metrics || {};
                document.getElementById("latest-epoch").textContent = safe(latest.epoch);
                document.getElementById("latest-val-acc").textContent = formatNum(latestMetrics.val_acc, 2, "%");
                document.getElementById("latest-val-f1").textContent = formatNum(latestMetrics.val_f1, 4, "");
                document.getElementById("latest-phase").textContent = safe(latest.phase);

                renderChart(data.history.train_acc || [], data.history.val_acc || []);
                document.getElementById("status").textContent = `Last updated: ${data.updated_at}`;
            } catch (err) {
                document.getElementById("status").textContent = `Last updated: error (${err})`;
            }
        }

        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


def _bgr_to_b64(img_bgr: np.ndarray) -> str:
    """Encode a BGR uint8 numpy image to base64 PNG string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None or _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb = np.array(pil)
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Run preprocessing pipeline to get intermediate images
        pipeline_images: dict[str, str] = {}
        try:
            from src.preprocessing import run_full_pipeline
            pipe = run_full_pipeline(frame_bgr)
            pipeline_images = {
                "original":       _bgr_to_b64(pipe["original"]),
                "step1_resized":  _bgr_to_b64(pipe["step1_resized"]),
                "step2_denoised": _bgr_to_b64(pipe["step2_denoised"]),
                "step3_lab":      _bgr_to_b64(pipe["step3_lab"]),
                "step4_clahe":    _bgr_to_b64(pipe["step4_clahe"]),
                "step5_no_hair":  _bgr_to_b64(pipe["step5_no_hair"]),
            }
        except Exception:
            pass  # pipeline images are optional — prediction still works

        result = _detector.predict_frame(frame_bgr)
        cls_name = str(result["class"])
        class_idx = int(CLASS_NAMES.index(cls_name)) if cls_name in CLASS_NAMES else -1

        return {
            "status": "success",
            "prediction": PREDICTION_LABELS.get(cls_name, cls_name.lower()),
            "class_index": class_idx,
            "confidence": float(result["confidence"]),
            "is_uncertain": bool(result["is_uncertain"]),
            "probabilities": {k: float(v) for k, v in result["probabilities"].items()},
            "preprocessing_applied": True,
            "pipeline_images": pipeline_images,
            "gradcam": _bgr_to_b64(result["gradcam_overlay_bgr"]) if result.get("gradcam_overlay_bgr") is not None else "",
        }

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
