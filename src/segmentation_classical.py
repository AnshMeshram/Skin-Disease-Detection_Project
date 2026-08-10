from pathlib import Path
from typing import Any
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_bgr(img_bgr: np.ndarray) -> None:
    assert isinstance(img_bgr, np.ndarray), "Input must be numpy array"
    assert img_bgr.dtype == np.uint8, f"Expected uint8, got {img_bgr.dtype}"
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, f"Expected (H,W,3), got {img_bgr.shape}"


def _assert_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    assert isinstance(mask, np.ndarray), "Mask must be numpy array"
    assert mask.dtype == np.uint8, f"Expected uint8, got {mask.dtype}"
    assert mask.shape == shape, f"Expected {shape}, got {mask.shape}"
    vals = np.unique(mask)
    assert set(vals.tolist()).issubset({0, 255}), f"Mask must be binary {{0,255}}, got {vals}"


def _morph_cleanup(mask: np.ndarray) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_open, iterations=2)
    return np.where(out > 0, 255, 0).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if not contours:
        return out
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, [largest], -1, 255, thickness=-1)
    return out


def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """Otsu thresholding with morphological refinement and largest component fill."""
    _assert_bgr(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = _morph_cleanup(mask)
    mask = _largest_component(mask)
    _assert_mask(mask, img_bgr.shape[:2])
    return mask


def segment_watershed(img_bgr: np.ndarray) -> np.ndarray:
    """Watershed-based lesion segmentation with marker construction."""
    _assert_bgr(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)[1].astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr.copy(), markers)

    lesion = np.zeros(gray.shape, dtype=np.uint8)
    lesion[markers > 1] = 255
    lesion = _morph_cleanup(lesion)
    lesion = _largest_component(lesion)

    _assert_mask(lesion, img_bgr.shape[:2])
    return lesion


def segment_grabcut(img_bgr: np.ndarray, rect_margin: float = 0.1) -> np.ndarray:
    """GrabCut segmentation initialized by center rectangle."""
    _assert_bgr(img_bgr)
    h, w = img_bgr.shape[:2]
    x = int(w * rect_margin)
    y = int(h * rect_margin)
    rw = max(1, int(w * (1.0 - 2 * rect_margin)))
    rh = max(1, int(h * (1.0 - 2 * rect_margin)))
    rect = (x, y, rw, rh)

    gcmask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr.copy(), gcmask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((gcmask == 2) | (gcmask == 0), 0, 255).astype(np.uint8)
    mask2 = _morph_cleanup(mask2)
    mask2 = _largest_component(mask2)
    _assert_mask(mask2, img_bgr.shape[:2])
    return mask2


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray, bg_color: tuple[int, int, int] = (0, 0, 0)) -> dict[str, Any]:
    """Apply binary mask and return masked image, lesion crop, and bbox."""
    _assert_bgr(img_bgr)
    _assert_mask(mask, img_bgr.shape[:2])

    masked = np.full_like(img_bgr, bg_color, dtype=np.uint8)
    masked[mask == 255] = img_bgr[mask == 255]

    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        bbox = (0, 0, 0, 0)
        cropped = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = (x1, y1, x2, y2)
        cropped = masked[y1 : y2 + 1, x1 : x2 + 1]

    return {"masked_full": masked, "cropped_lesion": cropped, "bbox": bbox}


def segment_ensemble(img_bgr: np.ndarray, method: str = "vote") -> np.ndarray:
    """Combine Otsu, Watershed and GrabCut by vote/union/intersection."""
    _assert_bgr(img_bgr)
    masks = [segment_otsu(img_bgr), segment_watershed(img_bgr), segment_grabcut(img_bgr)]
    stack = np.stack([(m > 0).astype(np.uint8) for m in masks], axis=0)

    if method == "vote":
        out = (stack.sum(axis=0) >= 2).astype(np.uint8)
    elif method == "union":
        out = (stack.sum(axis=0) >= 1).astype(np.uint8)
    elif method == "intersection":
        out = (stack.sum(axis=0) == 3).astype(np.uint8)
    else:
        raise ValueError("method must be one of: vote, union, intersection")

    out = (out * 255).astype(np.uint8)
    out = _morph_cleanup(out)
    out = _largest_component(out)
    _assert_mask(out, img_bgr.shape[:2])
    return out


def compute_mask_quality(mask: np.ndarray, img_bgr: np.ndarray) -> dict[str, Any]:
    """Compute lesion area, geometry, contour, and validity metrics."""
    _assert_bgr(img_bgr)
    _assert_mask(mask, img_bgr.shape[:2])

    h, w = mask.shape
    area = int(np.sum(mask > 0))
    area_pct = float(100.0 * area / (h * w))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    if not contours:
        return {
            "lesion_area_px": 0,
            "lesion_area_pct": 0.0,
            "aspect_ratio": 0.0,
            "circularity": 0.0,
            "is_valid": False,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "contour_count": 0,
        }

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    perimeter = max(cv2.arcLength(cnt, True), 1e-6)
    circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))

    m = cv2.moments(cnt)
    cx = float(m["m10"] / m["m00"]) if m["m00"] > 0 else float(x + bw / 2)
    cy = float(m["m01"] / m["m00"]) if m["m00"] > 0 else float(y + bh / 2)

    return {
        "lesion_area_px": area,
        "lesion_area_pct": area_pct,
        "aspect_ratio": float(bw / max(bh, 1)),
        "circularity": circularity,
        "is_valid": bool(5.0 <= area_pct <= 90.0),
        "centroid_x": cx,
        "centroid_y": cy,
        "contour_count": contour_count,
    }


def _overlay_contour(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def save_segmentation_comparison(img_bgr: np.ndarray, out_path: str) -> None:
    """Save 4-panel comparison and lesion-isolated output."""
    _assert_bgr(img_bgr)
    otsu = segment_otsu(img_bgr)
    ws = segment_watershed(img_bgr)
    gc = segment_grabcut(img_bgr)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, title, mask in zip(axes[1:], ["Otsu", "Watershed", "GrabCut"], [otsu, ws, gc]):
        ax.imshow(cv2.cvtColor(_overlay_contour(img_bgr, mask), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    masked = apply_mask(img_bgr, segment_ensemble(img_bgr))["masked_full"]
    cv2.imwrite(str(out_file.parent / "masked_lesion.png"), masked)


def analyse_dataset_segmentation(config: dict, sample_n: int = 200, out_dir: str = "outputs/segmentation_analysis") -> None:
    """Run segmentation analysis over random sample and write plots + reports."""
    csv_path = Path(config["data"]["isic_ground_truth"])
    img_dir = Path(config["data"]["isic_images_dir"])
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    class_cols = [c for c in ["MEL", "NV", "BCC", "AK", "AKIEC", "BKL", "DF", "VASC", "SCC"] if c in df.columns]
    if "AKIEC" in class_cols and "AK" not in class_cols:
        df = df.rename(columns={"AKIEC": "AK"})
        class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    df["class_name"] = df[class_cols].idxmax(axis=1)
    sampled = df.sample(min(sample_n, len(df)), random_state=42)

    rows = []
    examples = []
    for _, r in sampled.iterrows():
        img_path = img_dir / f"{r['image']}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        mask = segment_ensemble(img)
        q = compute_mask_quality(mask, img)
        q["image"] = r["image"]
        q["class_name"] = r["class_name"]
        rows.append(q)
        if len(examples) < 9:
            examples.append((img, segment_otsu(img), segment_watershed(img), segment_grabcut(img)))

    rep = pd.DataFrame(rows)
    rep.to_csv(out_root / "quality_report.csv", index=False)

    summary = rep.groupby("class_name").agg(["mean", "std", "min", "max"])
    with open(out_root / "quality_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary.to_string())

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, (img, o, w, g) in enumerate(examples[:9]):
        r = i // 3
        c = i % 3
        panel = np.hstack([
            cv2.cvtColor(_overlay_contour(img, o), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(_overlay_contour(img, w), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(_overlay_contour(img, g), cv2.COLOR_BGR2RGB),
        ])
        axes[r, c].imshow(panel)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_root / "method_comparison.png", dpi=300)
    plt.close(fig)

    if not rep.empty:
        plt.figure(figsize=(10, 5))
        for cls, sub in rep.groupby("class_name"):
            plt.hist(sub["lesion_area_pct"], bins=20, alpha=0.4, label=cls)
        plt.legend(ncol=3, fontsize=8)
        plt.xlabel("Lesion area %")
        plt.ylabel("Frequency")
        plt.title("Lesion area distribution per class")
        plt.tight_layout()
        plt.savefig(out_root / "area_distribution.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    img = cv2.imread("raw/ISIC_2019_Training_Input/ISIC_0000000.jpg")
    if img is None:
        raise FileNotFoundError("Could not read raw/ISIC_2019_Training_Input/ISIC_0000000.jpg")
    mask_otsu = segment_otsu(img)
    mask_ws = segment_watershed(img)
    mask_gc = segment_grabcut(img)
    mask_ens = segment_ensemble(img)
    q = compute_mask_quality(mask_ens, img)
    save_segmentation_comparison(img, "outputs/seg_test.png")
    print(f"Otsu area: {q['lesion_area_pct']:.1f}%  Circularity: {q['circularity']:.3f}")
    print("Classical segmentation PASSED")
