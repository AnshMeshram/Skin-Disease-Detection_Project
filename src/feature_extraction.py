from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

try:
    import timm
except Exception:
    timm = None

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except Exception:
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]


class CNNFeatureExtractor:
    def __init__(self, model_name: str = "efficientnet_b3", checkpoint_path: str | None = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        if timm is None:
            raise ImportError("timm is required for CNNFeatureExtractor")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)

        self.model.eval().to(self.device)
        self.feature_dim = getattr(self.model, "num_features", 1536)

    def extract_batch(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CNN features"):
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x = x.to(self.device, non_blocking=True)
                f = self.model(x)
                feats.append(f.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    def extract_single(self, img_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(self.device)
            f = self.model(x)
        return f.squeeze(0).detach().cpu().numpy()

    def save_features(self, features: np.ndarray, labels: np.ndarray, out_path: str) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out.with_suffix(".npy")), features)
        np.save(str(out.parent / "labels.npy"), labels)
        meta = pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
        meta.to_csv(out.parent / "metadata.csv", index=False)


def extract_lbp_features(img_gray: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    if local_binary_pattern is None:
        return np.zeros(26, dtype=np.float32)
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist.astype(np.float32)


def extract_glcm_features(img_gray: np.ndarray, distances: list[int] | None = None, angles: list[float] | None = None) -> np.ndarray:
    if graycomatrix is None or graycoprops is None:
        return np.zeros(12, dtype=np.float32)
    distances = distances or [1, 3]
    angles = angles or [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    q = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(q, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    vec = []
    for d_idx, _ in enumerate(distances):
        for p in props:
            values = graycoprops(glcm, p)[d_idx]
            vec.append(float(np.mean(values)))
    return np.array(vec, dtype=np.float32)


def extract_colour_features(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3
    if mask is None:
        mask_bin = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    else:
        mask_bin = (mask > 0).astype(np.uint8) * 255

    b, g, r = cv2.split(img_bgr)
    rgb_hist = []
    for ch in [r, g, b]:
        h = cv2.calcHist([ch], [0], mask_bin, [32], [0, 256]).flatten()
        h = h / max(h.sum(), 1e-6)
        rgb_hist.append(h)
    rgb_hist = np.concatenate(rgb_hist)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask_bin, [36], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask_bin, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask_bin, [32], [0, 256]).flatten()
    hsv_hist = np.concatenate([h_hist / max(h_hist.sum(), 1e-6), s_hist / max(s_hist.sum(), 1e-6), v_hist / max(v_hist.sum(), 1e-6)])

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_stats = []
    for i in range(3):
        vals = lab[:, :, i][mask_bin > 0]
        if len(vals) == 0:
            vals = np.array([0], dtype=np.float32)
        lab_stats += [float(vals.mean()), float(vals.std())]

    pixels = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[mask_bin > 0]
    if len(pixels) < 3:
        dom = np.zeros((3, 3), dtype=np.float32)
    else:
        km = cv2.kmeans(
            np.float32(pixels),
            3,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        dom = km[2]
    dom = dom.reshape(-1)

    mid = img_bgr.shape[1] // 2
    left = img_bgr[:, :mid]
    right = img_bgr[:, mid:]
    asym = []
    for ch_l, ch_r in zip(cv2.split(left), cv2.split(right)):
        hl = cv2.calcHist([ch_l], [0], None, [32], [0, 256]).flatten()
        hr = cv2.calcHist([ch_r], [0], None, [32], [0, 256]).flatten()
        hl = hl / max(hl.sum(), 1e-6)
        hr = hr / max(hr.sum(), 1e-6)
        asym.append(float(np.mean(np.abs(hl - hr))))

    vec = np.concatenate([rgb_hist, hsv_hist, np.array(lab_stats, dtype=np.float32), dom.astype(np.float32), np.array(asym, dtype=np.float32)])
    return vec.astype(np.float32)


def extract_shape_features(mask: np.ndarray) -> dict[str, Any]:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    area = float(np.sum(m > 0))
    area_pct = float(area / (h * w + 1e-6) * 100.0)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        base = {
            "asymmetry": 0.0,
            "border_irregularity": 0.0,
            "diameter_eq": 0.0,
            "lesion_area_px": 0.0,
            "lesion_area_pct": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "n_contours": 0.0,
            "circularity": 0.0,
        }
        base["vector"] = np.array(list(base.values()), dtype=np.float32)
        return base

    cnt = max(contours, key=cv2.contourArea)
    peri = max(cv2.arcLength(cnt, True), 1e-6)
    circularity = float((4 * np.pi * area) / (peri * peri))

    x, y, bw, bh = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1e-6)
    cnt_area = max(cv2.contourArea(cnt), 1e-6)

    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        asym = float(max(ma, mi) / max(min(ma, mi), 1e-6))
    else:
        asym = float(max(bw, bh) / max(min(bw, bh), 1e-6))

    mmt = cv2.moments(cnt)
    cx = float(mmt["m10"] / mmt["m00"]) / max(w, 1) if mmt["m00"] > 0 else 0.5
    cy = float(mmt["m01"] / mmt["m00"]) / max(h, 1) if mmt["m00"] > 0 else 0.5

    out = {
        "asymmetry": asym,
        "border_irregularity": float((peri * peri) / (4 * np.pi * cnt_area)),
        "diameter_eq": float(np.sqrt((4 * cnt_area) / np.pi)),
        "lesion_area_px": float(cnt_area),
        "lesion_area_pct": area_pct,
        "aspect_ratio": float(bw / max(bh, 1)),
        "solidity": float(cnt_area / hull_area),
        "extent": float(cnt_area / max(bw * bh, 1)),
        "centroid_x": cx,
        "centroid_y": cy,
        "n_contours": float(len(contours)),
        "circularity": circularity,
    }
    out["vector"] = np.array(list(out.values()), dtype=np.float32)
    return out


def extract_all_features(img_bgr: np.ndarray, mask: np.ndarray | None = None, cnn_extractor: CNNFeatureExtractor | None = None) -> dict[str, Any]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    deep = None
    if cnn_extractor is not None:
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (300, 300)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        deep = cnn_extractor.extract_single(t)

    lbp = extract_lbp_features(gray)
    glcm = extract_glcm_features(gray)
    colour = extract_colour_features(img_bgr, mask=mask)
    shape = extract_shape_features(mask if mask is not None else np.ones(gray.shape, dtype=np.uint8) * 255)

    handcrafted = np.concatenate([lbp, glcm, colour, shape["vector"]]).astype(np.float32)
    return {
        "deep": deep,
        "lbp": lbp,
        "glcm": glcm,
        "colour": colour,
        "shape": shape,
        "combined_handcrafted": handcrafted,
    }


def run_feature_analysis(config: dict, out_dir: str = "outputs/feature_analysis", sample_n: int = 500) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = Path(config["data"]["isic_ground_truth"])
    img_dir = Path(config["data"]["isic_images_dir"])
    df = pd.read_csv(csv_path)

    class_cols = [c for c in ["MEL", "NV", "BCC", "AK", "AKIEC", "BKL", "DF", "VASC", "SCC"] if c in df.columns]
    if "AKIEC" in class_cols and "AK" not in class_cols:
        df = df.rename(columns={"AKIEC": "AK"})
        class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    df["class_name"] = df[class_cols].idxmax(axis=1)
    sampled = df.groupby("class_name", group_keys=False).apply(lambda x: x.sample(min(max(1, sample_n // len(class_cols)), len(x)), random_state=42)).reset_index(drop=True)

    feats = []
    labels = []
    color_rows = []
    tex_rows = []
    abcd_rows = []

    for _, r in tqdm(sampled.iterrows(), total=len(sampled), desc="feature analysis"):
        p = img_dir / f"{r['image']}.jpg"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        f = extract_all_features(img)
        feats.append(f["combined_handcrafted"])
        labels.append(r["class_name"])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        color_rows.append({"class": r["class_name"], "L": lab[:, :, 0].mean(), "A": lab[:, :, 1].mean(), "B": lab[:, :, 2].mean()})

        tex_rows.append({
            "class": r["class_name"],
            "contrast": f["glcm"][0],
            "dissimilarity": f["glcm"][1],
            "homogeneity": f["glcm"][2],
            "energy": f["glcm"][3],
            "correlation": f["glcm"][4],
            "ASM": f["glcm"][5],
        })

        abcd_rows.append({"class": r["class_name"], "asymmetry": f["shape"]["asymmetry"], "border": f["shape"]["border_irregularity"]})

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels)

    np.save(out / "features.npy", X)
    np.save(out / "labels.npy", y)

    if len(X) >= 10:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 5)), n_iter=1000, random_state=42)
        Z = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for c in np.unique(y):
            m = y == c
            plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=c)
        plt.legend(ncol=3, fontsize=8)
        plt.title("t-SNE deep features")
        plt.tight_layout()
        plt.savefig(out / "tsne_deep_features.png", dpi=300)
        plt.close()

    try:
        import umap

        reducer = umap.UMAP(random_state=42)
        U = reducer.fit_transform(X)
        plt.figure(figsize=(8, 6))
        for c in np.unique(y):
            m = y == c
            plt.scatter(U[m, 0], U[m, 1], s=12, alpha=0.7, label=c)
        plt.legend(ncol=3, fontsize=8)
        plt.title("UMAP deep features")
        plt.tight_layout()
        plt.savefig(out / "umap_deep_features.png", dpi=300)
        plt.close()
    except Exception:
        pass

    if len(np.unique(y)) > 1 and len(X) > 10:
        y_num = pd.Categorical(y).codes
        fvals, _ = f_classif(X, y_num)
        top = np.argsort(np.nan_to_num(fvals))[-20:]
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(len(top)), fvals[top])
        plt.xticks(np.arange(len(top)), [f"f{i}" for i in top], rotation=60)
        plt.title("Top-20 handcrafted features by ANOVA F")
        plt.tight_layout()
        plt.savefig(out / "feature_importance.png", dpi=300)
        plt.close()

    cdf = pd.DataFrame(color_rows)
    if not cdf.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, ch in enumerate(["L", "A", "B"]):
            sns.boxplot(data=cdf, x="class", y=ch, ax=axes[i])
            axes[i].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "colour_distribution_per_class.png", dpi=300)
        plt.close(fig)

    tdf = pd.DataFrame(tex_rows)
    if not tdf.empty:
        means = tdf.groupby("class").mean()
        means = pd.DataFrame(MinMaxScaler().fit_transform(means), index=means.index, columns=means.columns)
        plt.figure(figsize=(8, 5))
        sns.heatmap(means, cmap="viridis", annot=True, fmt=".2f")
        plt.title("Texture heatmap")
        plt.tight_layout()
        plt.savefig(out / "texture_heatmap.png", dpi=300)
        plt.close()

    adf = pd.DataFrame(abcd_rows)
    if not adf.empty:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=adf, x="asymmetry", y="border", hue="class", s=30)
        plt.tight_layout()
        plt.savefig(out / "abcd_scatter.png", dpi=300)
        plt.close()

    if len(X) > 10:
        idx = np.argsort(np.nan_to_num(np.var(X, axis=0)))[-50:]
        corr = np.corrcoef(X[:, idx], rowvar=False)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature correlation top-50")
        plt.tight_layout()
        plt.savefig(out / "feature_correlation.png", dpi=300)
        plt.close()

    if len(X) > 0:
        all_df = pd.DataFrame(X)
        all_df["class"] = y
        summary = all_df.groupby("class").agg(["mean", "std"])
        summary.to_csv(out / "feature_summary.csv")


if __name__ == "__main__":
    img = cv2.imread("raw/ISIC_2019_Training_Input/ISIC_0000000.jpg")
    if img is None:
        raise FileNotFoundError("Could not read raw/ISIC_2019_Training_Input/ISIC_0000000.jpg")
    features = extract_all_features(img)
    print(f"LBP features:    {features['lbp'].shape}")
    print(f"GLCM features:   {features['glcm'].shape}")
    print(f"Colour features: {features['colour'].shape}")
    print(f"Shape (ABCD):    {features['shape']}")
    print(f"Combined vector: {features['combined_handcrafted'].shape}")
    print("Feature extraction PASSED")
