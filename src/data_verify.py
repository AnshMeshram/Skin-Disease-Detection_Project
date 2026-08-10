from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "raw/ISIC_2019_Training_GroundTruth.csv"
ISIC_DIR = PROJECT_ROOT / "raw/ISIC_2019_Training_Input"
HEALTHY_DIR = PROJECT_ROOT / "raw/healthy_skin"
PLOT_PATH = PROJECT_ROOT / "outputs/plots/dataset_distribution.png"

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_class_columns(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    print("CSV columns:", columns)

    preferred = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    fallback = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]

    if all(col in columns for col in preferred):
        return preferred
    if all(col in columns for col in fallback):
        return fallback

    missing_preferred = [c for c in preferred if c not in columns]
    missing_fallback = [c for c in fallback if c not in columns]
    raise ValueError(
        "Could not detect valid class columns. "
        f"Missing for AK set: {missing_preferred}; "
        f"Missing for AKIEC set: {missing_fallback}"
    )


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_healthy_dir() -> Path:
    if HEALTHY_DIR.exists():
        return HEALTHY_DIR

    raw_root = PROJECT_ROOT / "raw"
    if not raw_root.exists():
        return HEALTHY_DIR

    candidates = [
        p
        for p in raw_root.iterdir()
        if p.is_dir() and "healthy" in p.name.lower()
    ]
    if not candidates:
        return HEALTHY_DIR

    candidates.sort(key=lambda p: len(list_images(p)), reverse=True)
    return candidates[0]


def print_distribution(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    print("\nClass distribution:")
    print(f"{'Class':<10} {'Count':>8} {'% of total':>12}")
    print("-" * 32)
    for class_name in CLASS_NAMES:
        count = counts[class_name]
        pct = (count / total * 100.0) if total > 0 else 0.0
        print(f"{class_name:<10} {count:>8} {pct:>11.2f}%")


def save_plot(counts: dict[str, int], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    classes = CLASS_NAMES
    values = [counts[c] for c in classes]

    plt.figure(figsize=(11, 5))
    bars = plt.bar(classes, values, color="#4C72B0")
    plt.title("Dataset Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.xticks(rotation=30)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {CSV_PATH}")

    header_df = pd.read_csv(CSV_PATH, nrows=0)
    detected_columns = detect_class_columns(header_df)

    full_df = pd.read_csv(CSV_PATH)
    isic_total_files = sum(1 for p in ISIC_DIR.iterdir() if p.is_file()) if ISIC_DIR.exists() else 0
    isic_images = list_images(ISIC_DIR)
    isic_total_on_disk = len(isic_images)
    csv_total = len(full_df)
    healthy_dir = resolve_healthy_dir()
    print(f"Healthy directory: {healthy_dir}")
    print(f"ISIC total files in folder: {isic_total_files}")
    print(f"ISIC images on disk: {isic_total_on_disk}")
    print(f"ISIC CSV rows: {csv_total}")

    display_to_csv = {
        "MEL": detected_columns[0],
        "NV": detected_columns[1],
        "BCC": detected_columns[2],
        "AK": detected_columns[3],
        "BKL": detected_columns[4],
        "DF": detected_columns[5],
        "VASC": detected_columns[6],
        "SCC": detected_columns[7],
    }

    class_counts = {
        "MEL": int(full_df[display_to_csv["MEL"]].sum()),
        "NV": int(full_df[display_to_csv["NV"]].sum()),
        "BCC": int(full_df[display_to_csv["BCC"]].sum()),
        "AK": int(full_df[display_to_csv["AK"]].sum()),
        "BKL": int(full_df[display_to_csv["BKL"]].sum()),
        "DF": int(full_df[display_to_csv["DF"]].sum()),
        "VASC": int(full_df[display_to_csv["VASC"]].sum()),
        "SCC": int(full_df[display_to_csv["SCC"]].sum()),
        "Healthy": len(list_images(healthy_dir)),
    }

    print_distribution(class_counts)

    missing_classes = [name for name, count in class_counts.items() if count <= 0]
    if missing_classes:
        print(f"\nCRITICAL error: classes with zero samples: {missing_classes}")
    else:
        print("\nAll 9 classes have samples (> 0).")

    healthy_images = list_images(healthy_dir)
    sample_paths = (isic_images + healthy_images)[:200]

    bad = []
    for img_path in sample_paths:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            bad.append(str(img_path))

    print(f"Corrupted images found: {len(bad)}")

    save_plot(class_counts, PLOT_PATH)
    print(f"Saved plot: {PLOT_PATH}")

    total_images = sum(class_counts.values())
    unmatched_isic = isic_total_on_disk - csv_total
    status = "READY TO TRAIN"
    if missing_classes or bad:
        status = "CRITICAL ERROR"

    print("\nFinal summary")
    print(f"ISIC_TOTAL_FILES: {isic_total_files}")
    print(f"ISIC_TOTAL_ON_DISK: {isic_total_on_disk}")
    print(f"ISIC_TOTAL_IN_CSV: {csv_total}")
    print(f"UNANNOTATED_ISIC_FILES: {unmatched_isic}")
    print(f"TOTAL: {total_images} images across 9 classes")
    print(f"MISSING classes: {missing_classes}")
    print(f"CORRUPTED: {len(bad)}")
    print(f"STATUS: {status}")


if __name__ == "__main__":
    main()
