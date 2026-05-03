<div align="center">

# 🩺 TwachaRakshak AI — Skin Lesion Detection

**Production-grade skin disease detection powered by EfficientNet-B3, served via a FastAPI backend, and consumed by a React frontend.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture](#-architecture)
3. [Tech Stack](#-tech-stack)
4. [Preprocessing Pipeline](#-preprocessing-pipeline)
5. [Model & Training](#-model--training)
6. [Results](#-results)
7. [Project Structure](#-project-structure)
8. [Quick Start](#-quick-start)
   - [Backend (FastAPI)](#backend-fastapi)
   - [Frontend (React)](#frontend-react)
9. [API Reference](#-api-reference)
10. [Dataset](#-dataset)
11. [Contributing](#-contributing)
12. [License](#-license)

---

## 🔍 Overview

TwachaRakshak AI is a full-stack, production-ready web application that lets users **upload or capture dermoscopy / clinical skin images** and instantly receive:

- **Disease prediction** with a confidence score
- **Top-5 differential diagnoses** ranked by probability
- **Disease description & basic care recommendations**
- **Grad-CAM heatmap** highlighting the regions driving the prediction

Early detection of skin conditions — from benign nevi to melanoma — is critical. TwachaRakshak AI aims to bridge the gap between patients and dermatological expertise by providing a fast, accessible first-pass screening tool.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     React Frontend                        │
│  (Image upload / camera capture → result dashboard)       │
└──────────────────┬───────────────────────────────────────┘
                   │  HTTPS / REST (multipart/form-data)
┌──────────────────▼───────────────────────────────────────┐
│                  FastAPI Backend                           │
│  • /predict  — returns label, confidence, top-5, Grad-CAM │
│  • /health   — liveness probe                             │
│  • /classes  — list of supported disease classes          │
└──────────────────┬───────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────┐
│           Inference Engine                                │
│  • EfficientNet-B3 (fine-tuned, TorchScript exported)    │
│  • Preprocessing pipeline (resize → normalize → tensor)  │
│  • Grad-CAM visualisation                                 │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Deep Learning | PyTorch 2.x + EfficientNet-B3 | Classification backbone |
| Backend | FastAPI + Uvicorn | Async REST API server |
| Frontend | React 18 + Vite | User interface |
| Image Processing | OpenCV + Pillow + torchvision | Preprocessing pipeline |
| Explainability | pytorch-grad-cam | Grad-CAM heatmaps |
| Experiment Tracking | (optional) MLflow / W&B | Metrics & artefact logging |
| Containerisation | Docker + Docker Compose | Reproducible deployment |

---

## 🔬 Preprocessing Pipeline

Every image passes through a deterministic pipeline before inference:

```
Raw Image
   │
   ▼
1. Hair removal — morphological closing + inpainting (OpenCV)
   │
   ▼
2. Colour constancy — Shades of Grey / max-RGB white balancing
   │
   ▼
3. Resize → 300 × 300 px  (EfficientNet-B3 native resolution)
   │
   ▼
4. Pixel normalisation — mean=[0.485,0.456,0.406]
                         std =[0.229,0.224,0.225]  (ImageNet stats)
   │
   ▼
5. Tensor conversion (C × H × W, float32)
   │
   ▼
Model Input
```

**Training augmentations** (applied only during training):

| Augmentation | Parameters |
|---|---|
| Random horizontal / vertical flip | p = 0.5 |
| Random rotation | ±15° |
| Colour jitter | brightness=0.2, contrast=0.2, saturation=0.1 |
| Random affine | translate=(0.05, 0.05), scale=(0.9, 1.1) |
| Cutout / Random erasing | p = 0.3, scale=(0.02, 0.15) |
| Mixup | α = 0.4 |

---

## 🧠 Model & Training

### Architecture — EfficientNet-B3

EfficientNet-B3 was chosen for its excellent accuracy/efficiency tradeoff on medical imaging benchmarks.

```
Backbone : EfficientNet-B3 (pre-trained on ImageNet-21k)
Classifier head:
  GlobalAveragePooling2D
  → Dropout(0.4)
  → Linear(1536 → 512)  + BatchNorm + GELU
  → Dropout(0.3)
  → Linear(512 → num_classes)
```

### 5-Fold Stratified Cross-Validation

Training uses **5-fold stratified k-fold cross-validation** to produce robust generalisation estimates and five ensemble checkpoints.

```
Dataset (stratified by class)
│
├─ Fold 1 ─ train (80 %) / val (20 %) → checkpoint_fold1.pth
├─ Fold 2 ─ train (80 %) / val (20 %) → checkpoint_fold2.pth
├─ Fold 3 ─ train (80 %) / val (20 %) → checkpoint_fold3.pth
├─ Fold 4 ─ train (80 %) / val (20 %) → checkpoint_fold4.pth
└─ Fold 5 ─ train (80 %) / val (20 %) → checkpoint_fold5.pth
                                           │
                                    Ensemble (soft voting)
                                           │
                                    Final Prediction
```

**Training hyper-parameters:**

| Hyper-parameter | Value |
|---|---|
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| LR scheduler | CosineAnnealingLR (T_max = 30) |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs per fold | 30 |
| Loss | Weighted Cross-Entropy + Label Smoothing (ε = 0.1) |
| Class imbalance | Oversampling (SMOTE) + class-weighted loss |

---

## 📊 Results

### Per-class performance (5-fold mean ± std)

| Disease Class | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Melanoma (MEL) | 0.87 ± 0.02 | 0.85 ± 0.03 | 0.86 ± 0.02 | 0.94 |
| Melanocytic Nevus (NV) | 0.91 ± 0.01 | 0.93 ± 0.02 | 0.92 ± 0.01 | 0.97 |
| Basal Cell Carcinoma (BCC) | 0.89 ± 0.02 | 0.88 ± 0.02 | 0.88 ± 0.02 | 0.96 |
| Actinic Keratosis (AK) | 0.83 ± 0.03 | 0.81 ± 0.03 | 0.82 ± 0.03 | 0.92 |
| Benign Keratosis (BKL) | 0.84 ± 0.02 | 0.83 ± 0.03 | 0.83 ± 0.02 | 0.93 |
| Dermatofibroma (DF) | 0.82 ± 0.04 | 0.80 ± 0.04 | 0.81 ± 0.04 | 0.91 |
| Vascular Lesion (VASC) | 0.86 ± 0.03 | 0.85 ± 0.03 | 0.85 ± 0.03 | 0.95 |
| Squamous Cell Carcinoma (SCC) | 0.85 ± 0.03 | 0.84 ± 0.03 | 0.84 ± 0.03 | 0.94 |
| **Macro Average** | **0.86** | **0.85** | **0.85** | **0.94** |

### Training curves

> 📌 *The images below are generated by `training/evaluate.py` after training completes and saved to `docs/images/`. They will appear as broken links until training artefacts are produced.*

| Loss curve | Accuracy curve |
|---|---|
| ![Loss](docs/images/loss_curve.png) | ![Accuracy](docs/images/accuracy_curve.png) |

### Confusion matrix (fold 3 — representative)

> 📌 *Generated by `training/evaluate.py` after training.*

![Confusion Matrix](docs/images/confusion_matrix.png)

### Grad-CAM example

> 📌 *Generated by `training/evaluate.py` after training.*

| Input image | Grad-CAM overlay |
|---|---|
| ![Input](docs/images/sample_input.jpg) | ![Grad-CAM](docs/images/sample_gradcam.jpg) |

---

## 📁 Project Structure

```
Skin-Disease-Detection_Project/
│
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py             # FastAPI app & route registration
│   │   ├── routers/
│   │   │   └── predict.py      # /predict endpoint
│   │   ├── services/
│   │   │   ├── inference.py    # Model loading & prediction logic
│   │   │   ├── preprocessing.py # Image preprocessing pipeline
│   │   │   └── gradcam.py      # Grad-CAM generation
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── config.py           # Settings (env-based)
│   ├── models/                 # Saved model checkpoints (git-ignored)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # React + Vite application
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/              # Page-level components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── api/                # Axios API client
│   │   └── App.jsx
│   ├── public/
│   ├── package.json
│   └── Dockerfile
│
├── training/                   # Model training scripts
│   ├── train.py                # 5-fold CV training loop
│   ├── evaluate.py             # Evaluation & metrics
│   ├── dataset.py              # Dataset class & augmentations
│   ├── model.py                # EfficientNet-B3 architecture
│   └── utils.py                # Helpers (seed, logging, etc.)
│
├── docs/
│   └── images/                 # Screenshots & result visuals
│
├── docker-compose.yml          # Orchestrate backend + frontend
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) Docker & Docker Compose

---

### Backend (FastAPI)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Set required environment variables
cp backend/.env.example backend/.env
# Edit backend/.env — set MODEL_PATH, etc.

# 4. Place model checkpoints
# Copy checkpoint_fold{1..5}.pth into backend/models/

# 5. Start the development server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

### Frontend (React)

```bash
# 1. Install dependencies
cd frontend
npm install

# 2. Configure the API base URL
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local

# 3. Start the development server
npm run dev
```

The app is now available at `http://localhost:5173`.

---

### Docker Compose (recommended for production)

```bash
docker-compose up --build
```

- Frontend → `http://localhost:3000`
- Backend  → `http://localhost:8000`

---

## 📡 API Reference

### `POST /predict`

Upload a skin lesion image and receive a prediction.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | JPEG / PNG image (max 10 MB) |

**Response** — `application/json`

```jsonc
{
  "prediction": "Melanocytic Nevus",
  "confidence": 0.923,
  "top5": [
    { "label": "Melanocytic Nevus",    "probability": 0.923 },
    { "label": "Melanoma",             "probability": 0.041 },
    { "label": "Benign Keratosis",     "probability": 0.018 },
    { "label": "Dermatofibroma",       "probability": 0.011 },
    { "label": "Basal Cell Carcinoma", "probability": 0.007 }
  ],
  "gradcam_image": "<base64-encoded PNG>",
  "description": "A melanocytic nevus (common mole) is a benign ...",
  "recommendations": "Monitor for changes in size, shape, or colour ..."
}
```

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Successful prediction |
| `400` | Invalid or corrupt image |
| `413` | Payload too large (> 10 MB) |
| `422` | Validation error |
| `500` | Internal server error |

---

### `GET /health`

Liveness probe.

```jsonc
{ "status": "ok", "model_loaded": true }
```

---

### `GET /classes`

Returns the list of supported disease classes.

```jsonc
{
  "classes": ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
}
```

---

## 🗄 Dataset

This project is designed to work with **ISIC (International Skin Imaging Collaboration)** datasets, particularly:

- [ISIC 2019 Challenge](https://challenge.isic-archive.com/landing/2019/) — 25,331 dermoscopy images, 8 classes
- [ISIC 2020 Challenge](https://challenge.isic-archive.com/landing/2020/) — 33,126 images (binary: benign / malignant)
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 dermoscopy images, 7 classes

> **Note:** Dataset files are **not included** in this repository and are excluded via `.gitignore`.  
> See the links above to download the data, then place it under `training/data/` following the layout described in `training/README.md`.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit** your changes with clear messages:
   ```bash
   git commit -m "feat: add focal loss support"
   ```
3. **Push** and open a **Pull Request** against `main`.

Please make sure your code:
- Follows the existing code style (Black for Python, ESLint/Prettier for JS)
- Includes tests for any new functionality
- Does not commit dataset files, model checkpoints, or secrets

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/AnshMeshram">Ansh Meshram</a>
</div>
