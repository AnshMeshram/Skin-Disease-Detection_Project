NUM_CLASSES = 9

import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


DISEASE_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = DISEASE_CLASSES + ["Healthy"]


def detect_disease_columns(csv_path: str) -> list[str]:
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    candidates_ak = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    candidates_akiec = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
    if all(c in cols for c in candidates_ak):
        print("  CSV: AK column variant detected")
        return candidates_ak
    if all(c in cols for c in candidates_akiec):
        print("  CSV: AKIEC column variant detected")
        return candidates_akiec
    raise ValueError(f"Cannot detect disease columns. CSV has: {cols}")


def _resolve_config_paths(config: dict) -> tuple[Path, Path, Path, Path, int]:
    data_cfg = config.get("data", {})
    csv_path = Path(data_cfg.get("isic_ground_truth", config.get("isic_ground_truth", "raw/ISIC_2019_Training_GroundTruth.csv")))
    isic_dir = Path(data_cfg.get("isic_images_dir", config.get("isic_images_dir", "raw/ISIC_2019_Training_Input")))
    healthy_dir = Path(data_cfg.get("healthy_dir", config.get("healthy_dir", "raw/healthy_skin")))
    output_dir = Path(data_cfg.get("output_dir", config.get("output_dir", "outputs")))
    healthy_limit = int(data_cfg.get("healthy_limit", config.get("healthy_limit", 1000)))

    if not healthy_dir.exists():
        raw_root = Path("raw")
        if raw_root.exists():
            candidates = [p for p in raw_root.iterdir() if p.is_dir() and "healthy" in p.name.lower()]
            if candidates:
                healthy_dir = candidates[0]

    return csv_path, isic_dir, healthy_dir, output_dir, healthy_limit


def _print_distribution(labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=9)
    total = max(int(counts.sum()), 1)
    print("\nClass distribution")
    print(f"{'Class':<10} {'Count':>8} {'%':>8}")
    print("-" * 28)
    for i, class_name in enumerate(CLASS_NAMES):
        pct = (counts[i] / total) * 100.0
        print(f"{class_name:<10} {int(counts[i]):>8} {pct:>7.2f}")


def build_dataframe(config: dict, prefer_preprocessed: bool = False) -> pd.DataFrame:
    csv_path, isic_dir, healthy_dir, output_dir, healthy_limit = _resolve_config_paths(config)
    disease_cols = detect_disease_columns(str(csv_path))

    df = pd.read_csv(csv_path)
    df["disease_name"] = df[disease_cols].idxmax(axis=1)
    if "AKIEC" in disease_cols:
        df["disease_name"] = df["disease_name"].replace({"AKIEC": "AK"})

    label_map = {c: i for i, c in enumerate(DISEASE_CLASSES)}
    df["label"] = df["disease_name"].map(label_map).astype(int)
    df["image_path"] = df["image"].apply(lambda x: str(isic_dir / f"{x}.jpg"))
    df = df[["image", "image_path", "label"]].copy()

    healthy_paths = sorted([p for p in healthy_dir.rglob("*.jpg") if p.is_file()])[:healthy_limit]
    if healthy_paths:
        hdf = pd.DataFrame(
            {
                "image": [p.stem for p in healthy_paths],
                "image_path": [str(p) for p in healthy_paths],
                "label": 8,
            }
        )
        df = pd.concat([df, hdf], ignore_index=True)

    df["processed_path"] = ""
    df["mask_path"] = ""

    manifest = output_dir / "preprocessed/manifest.csv"
    if prefer_preprocessed and manifest.exists():
        man = pd.read_csv(manifest)
        keep = [c for c in ["image", "processed_path", "mask_path"] if c in man.columns]
        man = man[keep]
        df = df.merge(man, on="image", how="left", suffixes=("", "_m"))

        if "processed_path_m" in df.columns:
            df["processed_path"] = df["processed_path_m"].fillna("")
            df = df.drop(columns=["processed_path_m"])
        if "mask_path_m" in df.columns:
            df["mask_path"] = df["mask_path_m"].fillna("")
            df = df.drop(columns=["mask_path_m"])

        exists = df["processed_path"].apply(lambda p: bool(p) and Path(str(p)).exists())
        print(f"Using preprocessed images: {int(exists.sum())}/25331 ({100.0*exists.mean():.1f}%)")

    labels = df["label"].to_numpy(dtype=int)
    assert labels.min() >= 0 and labels.max() <= 8, "Found out-of-range labels"
    present = set(np.unique(labels).tolist())
    missing = [i for i in range(9) if i not in present]
    assert not missing, f"Missing classes: {[CLASS_NAMES[i] for i in missing]}"

    _print_distribution(labels)
    return df[["image", "image_path", "label", "processed_path", "mask_path"]]


def get_albumentations_train(config: dict):
    assert A is not None and ToTensorV2 is not None, "albumentations is required"
    image_size = int(config["image_size"])

    # Supports both albumentations v1 and v2 signatures.
    random_resized_crop = cast(Any, A.RandomResizedCrop)
    try:
        random_crop = random_resized_crop(
            size=(image_size, image_size),
            scale=(1 - float(config["zoom"]), 1.0),
            p=0.5,
        )
    except TypeError:
        random_crop = random_resized_crop(
            image_size,
            image_size,
            scale=(1 - float(config["zoom"]), 1.0),
            p=0.5,
        )

    coarse_dropout = cast(Any, A.CoarseDropout)
    try:
        coarse = coarse_dropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 20),
            hole_width_range=(8, 20),
            p=float(config["rand_erasing_prob"]),
        )
    except TypeError:
        coarse = coarse_dropout(
            max_holes=8,
            max_height=20,
            max_width=20,
            p=float(config["rand_erasing_prob"]),
        )

    return A.Compose(
        cast(
            list[Any],
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=float(config["flip_prob"])),
                A.VerticalFlip(p=float(config["flip_prob"])),
                A.Rotate(limit=float(config["rotation_degrees"]), p=0.7),
                A.ColorJitter(
                    brightness=float(config["color_jitter"]),
                    contrast=float(config["color_jitter"]),
                    saturation=float(config["color_jitter"]),
                    hue=float(config["color_jitter"]) / 2.0,
                    p=0.7,
                ),
                random_crop,
                coarse,
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
        )
    )


def get_albumentations_val(config: dict):
    assert A is not None and ToTensorV2 is not None, "albumentations is required"
    image_size = int(config["image_size"])
    return A.Compose(
        cast(
            list[Any],
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
        )
    )


class SkinLesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any = None, preprocess_fn: Any = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.df)

    def _safe_read(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        # Corrupted image fallback: neutral gray image.
        return np.full((300, 300, 3), 127, dtype=np.uint8)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        processed_path = str(row.get("processed_path", "") or "")
        if self.preprocess_fn is None and processed_path and Path(processed_path).exists():
            img_bgr = self._safe_read(processed_path)
        else:
            img_bgr = self._safe_read(img_path)
            if self.preprocess_fn is not None:
                img_bgr = self.preprocess_fn(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img_tensor = self.transform(image=img_rgb)["image"]
        else:
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return img_tensor, label


def mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[idx]
    return mixed, targets, targets[idx], float(lam)


def get_weighted_sampler(labels: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=9).astype(float)
    counts_safe = np.where(counts == 0, 1.0, counts)
    w = 1.0 / counts_safe
    w = np.clip(w / (w.mean() + 1e-12), 0.01, 1000.0)
    sw = np.where(np.isfinite(w[labels]), w[labels], 1.0)
    print(f"  Sampler weights: {{ {', '.join([f'{CLASS_NAMES[i]}: {round(float(w[i]),2)}' for i in range(9)])} }}")
    return WeightedRandomSampler(sw.tolist(), len(sw), replacement=True), w


def _loader_kwargs(config: dict, is_val: bool) -> dict:
    key = "val_num_workers" if is_val else "num_workers"
    nw = int(config.get(key, 0))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))
    persistent = bool(config.get("persistent_workers", False)) and nw > 0
    kwargs = {
        "num_workers": nw,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
    }
    if nw > 0:
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


def create_fold_dataloaders(df: pd.DataFrame, config: dict, fold_idx: int, preprocess_fn: Any = None):
    skf = StratifiedKFold(n_splits=int(config.get("folds", 5)), shuffle=True, random_state=int(config.get("seed", 42)))

    labels = df["label"].to_numpy(dtype=int)
    splits = list(skf.split(df, labels))
    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(f"fold_idx out of range: {fold_idx}")

    train_idx, val_idx = splits[fold_idx]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_tf = get_albumentations_train(config)
    val_tf = get_albumentations_val(config)

    train_ds = SkinLesionDataset(train_df, transform=train_tf, preprocess_fn=preprocess_fn)
    val_ds = SkinLesionDataset(val_df, transform=val_tf, preprocess_fn=None)

    sampler, class_weights = get_weighted_sampler(train_df["label"].to_numpy(dtype=int))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 32)),
        sampler=sampler,
        **_loader_kwargs(config, is_val=False),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        **_loader_kwargs(config, is_val=True),
    )

    return train_loader, val_loader, class_weights


def create_holdout_dataloaders(df: pd.DataFrame, config: dict, preprocess_fn: Any = None):
    """Create reproducible stratified 80/10/10 train/val/test dataloaders."""
    seed = int(config.get("seed", 42))

    split_cfg = config.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    labels = df["label"].to_numpy(dtype=int)
    idx_all = np.arange(len(df), dtype=int)

    # First split: train vs temp(val+test).
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        idx_all,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    # Second split: val vs test from temp.
    temp_labels = labels[temp_idx]
    val_from_temp = val_ratio / max(temp_ratio, 1e-12)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_from_temp,
        random_state=seed,
        shuffle=True,
        stratify=temp_labels,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_tf = get_albumentations_train(config)
    val_tf = get_albumentations_val(config)

    train_ds = SkinLesionDataset(train_df, transform=train_tf, preprocess_fn=preprocess_fn)
    val_ds = SkinLesionDataset(val_df, transform=val_tf, preprocess_fn=None)
    test_ds = SkinLesionDataset(test_df, transform=val_tf, preprocess_fn=None)

    sampler, class_weights = get_weighted_sampler(train_df["label"].to_numpy(dtype=int))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 32)),
        sampler=sampler,
        **_loader_kwargs(config, is_val=False),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        **_loader_kwargs(config, is_val=True),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        **_loader_kwargs(config, is_val=True),
    )

    print(
        f"Holdout split -> train:{len(train_df)} ({100.0*len(train_df)/len(df):.1f}%), "
        f"val:{len(val_df)} ({100.0*len(val_df)/len(df):.1f}%), "
        f"test:{len(test_df)} ({100.0*len(test_df)/len(df):.1f}%)"
    )
    return train_loader, val_loader, test_loader, class_weights
