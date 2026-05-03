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

<div align="center">

<b>Training & Evaluation</b><br>
<img src="docs/assets/graphs/loss_curves_all_folds.png" alt="Loss Curves (All Folds)" width="400"/>
<img src="docs/assets/loss_curve_mae.png" alt="Train/Validation Loss Curve" width="400"/>
<img src="docs/assets/graphs/confusion_matrix_final.png" alt="Confusion Matrix (Final)" width="400"/>
<img src="docs/assets/graphs/metrics_per_epoch_fold0.png" alt="Metrics per Epoch (Fold 0)" width="400"/>

<b>Feature Analysis</b><br>
<img src="docs/assets/graphs/tsne_embeddings.png" alt="t-SNE Embeddings" width="400"/>
<img src="docs/assets/graphs/feature_separability.png" alt="Feature Separability" width="400"/>
<img src="docs/assets/graphs/texture_heatmap.png" alt="Texture Heatmap" width="400"/>
<img src="docs/assets/graphs/colour_profile_per_class.png" alt="Colour Profile per Class" width="400"/>

<b>Dataset & Preprocessing</b><br>
<img src="docs/assets/graphs/class_distribution_bar.png" alt="Class Distribution (Bar)" width="400"/>
<img src="docs/assets/graphs/class_distribution_pie.png" alt="Class Distribution (Pie)" width="400"/>
<img src="docs/assets/graphs/image_size_distribution.png" alt="Image Size Distribution" width="400"/>
<img src="docs/assets/graphs/imbalance_comparison.png" alt="Imbalance Comparison" width="400"/>
<img src="docs/assets/graphs/sample_grid_per_class.png" alt="Sample Grid per Class" width="400"/>
<img src="docs/assets/graphs/pipeline_grid.png" alt="Preprocessing Pipeline Grid" width="400"/>
<img src="docs/assets/graphs/histogram_shift.png" alt="Histogram Shift" width="400"/>
<img src="docs/assets/graphs/hair_detection_samples.png" alt="Hair Detection Samples" width="400"/>
<img src="docs/assets/graphs/preprocessing_timing.png" alt="Preprocessing Timing" width="400"/>
<img src="docs/assets/graphs/preprocessing_samples.png" alt="Preprocessing Samples" width="400"/>

<b>Segmentation & Model Comparison</b><br>
<img src="docs/assets/graphs/abcd_scatter.png" alt="ABCD Scatter" width="400"/>
<img src="docs/assets/graphs/lesion_area_per_class.png" alt="Lesion Area per Class" width="400"/>
<img src="docs/assets/graphs/mask_quality_distribution.png" alt="Mask Quality Distribution" width="400"/>
<img src="docs/assets/graphs/method_comparison_grid.png" alt="Segmentation Method Comparison" width="400"/>
<img src="docs/assets/graphs/model_comparison.png" alt="Model Comparison" width="400"/>
<img src="docs/assets/graphs/dataset_distribution.png" alt="Dataset Distribution" width="400"/>
<img src="docs/assets/graphs/dataset_distribution_pie.png" alt="Dataset Distribution Pie" width="400"/>

</div>

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
