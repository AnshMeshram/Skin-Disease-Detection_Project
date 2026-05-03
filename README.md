# SkinAI - Skin Disease Detection (EfficientNet B3)

Clinical skin lesion classification across 9 classes with a FastAPI backend, a React frontend, and a k-fold training pipeline.

## Model In Use

- Architecture: EfficientNet B3
- Configured in: config.yaml
- Inference weights: outputs/efficientnet_b3/fold_0/best.pth

## Classes

MEL, NV, BCC, AK, BKL, DF, VASC, SCC, Healthy

## Methodology

1. Dataset curation from ISIC 2019 plus curated healthy samples
2. Preprocessing: hair removal, CLAHE, noise reduction, normalization
3. Training: 5-fold stratified CV, focal loss + label smoothing, cosine warm restarts, AMP
4. Evaluation: accuracy, balanced accuracy, macro F1, precision, recall

![Preprocessing Pipeline](docs/assets/pipeline_grid.png)

## Architecture

- Backbone: EfficientNet B3
- Head: dropout + linear classifier
- Input size: 192
- Classes: 9

## Results (Fold 4 Best)

From outputs/efficientnet_b3/fold_4/history.json:

| Metric              | Best Value |
| ------------------- | ---------- |
| Validation Accuracy | 90.65%     |
| Balanced Accuracy   | 95.48%     |
| Macro F1            | 0.9269     |
| Macro Precision     | 0.9023     |
| Specificity         | 0.9857     |

## Visuals

![Loss Curves (All Folds)](docs/assets/loss_curves_all_folds.png)
![Confusion Matrix (Final)](docs/assets/confusion_matrix_final.png)
![t-SNE Embeddings](docs/assets/tsne_embeddings.png)
![Model Comparison](docs/assets/model_comparison.png)
![Class Distribution](docs/assets/class_distribution_bar.png)

## Backend (FastAPI)

Start the API server:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Endpoints:

- GET /health
- POST /predict
- GET /metrics/latest

## Frontend (React + Vite)

```bash
cd skinai-frontend
npm install
npm run dev
```

Optional API override:

```bash
VITE_API_URL=http://localhost:8080
```

## Training and Evaluation

```bash
python main.py --stage verify   --config config.yaml
python main.py --stage train    --config config.yaml --model efficientnet_b3 --split_strategy kfold --fold 0
python main.py --stage evaluate --config config.yaml --model efficientnet_b3 --split_strategy kfold
python main.py --stage predict  --config config.yaml --model efficientnet_b3 --image path/to/image.jpg
```

## Project Structure

```
src/                 # training, inference, API
skinai-frontend/     # React UI
raw/                 # datasets (not in git)
outputs/             # generated artifacts (ignored)
docs/assets/         # tracked visuals for README
config.yaml          # training and inference config
```

## Copilot Jumpstart (Optional)

Prompt:
"Create a production-grade README for a skin lesion detection project with EfficientNet B3, FastAPI backend, React frontend, k-fold training, and embedded results visuals. Also add a repo-level gitignore for datasets, artifacts, and secrets."

## Safety and Limitations

This project is for research and educational use only and is not a medical device.
