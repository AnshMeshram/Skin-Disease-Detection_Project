import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_model_output_shape(config):
    from src.model_factory import build_model

    model = build_model("efficientnet_b3", num_classes=9, pretrained=False)
    out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 9), f"Wrong shape: {out.shape}"


def test_loss_not_nan(config):
    from src.dataset import get_weighted_sampler
    from src.losses import verify_loss

    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] * 5)
    _, class_weights = get_weighted_sampler(labels)
    assert verify_loss(config, class_weights, "cpu"), "Loss check failed"


def test_dataset_all_classes():
    from src.dataset import CLASS_NAMES, build_dataframe

    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    df = build_dataframe(cfg)
    for i in range(9):
        cnt = (df["label"] == i).sum()
        assert cnt > 0, f"Class {CLASS_NAMES[i]} has 0 samples!"


def test_csv_column_detection():
    from src.dataset import detect_disease_columns

    cols = detect_disease_columns("raw/ISIC_2019_Training_GroundTruth.csv")
    assert len(cols) == 8, f"Expected 8 disease columns, got {len(cols)}"
    assert "MEL" in cols and "NV" in cols


def test_sampler_no_nan():
    from src.dataset import get_weighted_sampler

    labels = np.array([0] * 1000 + [1] * 100 + [2] * 50 + [3] * 10 + [4] * 20 + [5] * 5 + [6] * 5 + [7] * 15 + [8] * 100)
    sampler, class_weights = get_weighted_sampler(labels)
    assert sampler is not None
    assert not np.any(np.isnan(class_weights)), "NaN in class weights"
    assert not np.any(np.isinf(class_weights)), "Inf in class weights"


def test_preprocessing_output_shape():
    from src.preprocessing import preprocess_image

    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.shape == (300, 300, 3)
    assert out.dtype == np.uint8


def test_metrics_all_present():
    from torch.utils.data import DataLoader, Dataset

    from src.evaluate import evaluate_model

    class TinyDataset(Dataset):
        def __len__(self):
            return 9

        def __getitem__(self, idx):
            x = torch.zeros(3, 224, 224)
            x[0, 0, 0] = float(idx)
            y = torch.tensor(idx, dtype=torch.long)
            return x, y

    class TinyModel(torch.nn.Module):
        def forward(self, inputs):
            batch = inputs.shape[0]
            logits = torch.full((batch, 9), -5.0)
            class_idx = inputs[:, 0, 0, 0].round().long().clamp(0, 8)
            logits[torch.arange(batch), class_idx] = 5.0
            return logits

    loader = DataLoader(TinyDataset(), batch_size=3, shuffle=False)
    model = TinyModel()
    metrics, _, _, _ = evaluate_model(model, loader, ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"])

    required = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "roc_auc_macro",
    ]
    for key in required:
        assert key in metrics, f"Missing metric: {key}"
