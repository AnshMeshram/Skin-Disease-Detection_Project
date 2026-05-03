import os
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ProcessPoolExecutor

# Import project modules
from src.feature_extraction import extract_all_features, CNNFeatureExtractor
from src.dataset import build_dataframe, CLASS_NAMES

class FeatureDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(row['image_path'])
        if img_bgr is None: img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (300, 300))
        
        img_tensor = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()

        return img_tensor, int(row['label']), img_resized, row['image_path']

def process_handcrafted_worker(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    f_hand = extract_all_features(img_bgr, cnn_extractor=None)
    
    # Also get RGB profile
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r_hist = cv2.calcHist([img_rgb], [0], None, [32], [0, 256]).flatten()
    g_hist = cv2.calcHist([img_rgb], [1], None, [32], [0, 256]).flatten()
    b_hist = cv2.calcHist([img_rgb], [2], None, [32], [0, 256]).flatten()
    
    return {
        'hand': f_hand['combined_handcrafted'],
        'glcm': f_hand['glcm'],
        'rgb': {
            'r': r_hist / (r_hist.sum() + 1e-6),
            'g': g_hist / (g_hist.sum() + 1e-6),
            'b': b_hist / (b_hist.sum() + 1e-6)
        }
    }

def generate_feature_report(config_path="config.yaml", out_dir="outputs/feature_report"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = build_dataframe(config)
    sampled_list = [df[df['label'] == lbl].sample(min(80, len(df[df['label'] == lbl])), random_state=42) for lbl in range(9)]
    sampled = pd.concat(sampled_list).reset_index(drop=True)
    print(f"Sampled {len(sampled)} images.")

    # 1. CNN Features (GPU - Fast)
    print("Step 1: Extracting deep features (GPU)...")
    dataset = FeatureDataset(sampled)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    extractor = CNNFeatureExtractor(model_name="efficientnet_b3", checkpoint_path="outputs/efficientnet_b3/fold_0/best.pth", device=str(device))
    all_deep = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls, _, _ in tqdm(loader, desc="CNN GPU"):
            feats = extractor.model(imgs.to(device))
            all_deep.append(feats.cpu().numpy())
            all_labels.extend(lbls.numpy())
            
    X_deep = np.concatenate(all_deep, axis=0)
    y = np.array(all_labels)

    # 2. Handcrafted Features (Parallel CPU - Fast)
    print("Step 2: Extracting handcrafted features (Parallel CPU)...")
    paths = sampled['image_path'].tolist()
    all_hand = []
    all_glcm = []
    all_rgb = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_handcrafted_worker, paths), total=len(paths), desc="Handcrafted CPU"))
        
    for i, res in enumerate(results):
        if res is None: continue
        lbl = y[i]
        all_hand.append(res['hand'])
        all_glcm.append({
            'class': CLASS_NAMES[lbl],
            'contrast': res['glcm'][0], 'dissimilarity': res['glcm'][1],
            'homogeneity': res['glcm'][2], 'energy': res['glcm'][3],
            'correlation': res['glcm'][4], 'ASM': res['glcm'][5]
        })
        all_rgb.append({'class': lbl, 'r': res['rgb']['r'], 'g': res['rgb']['g'], 'b': res['rgb']['b']})

    X_hand = np.array(all_hand)

    # 3. Visualization (Standard Matplotlib)
    print("Step 3: Generating Visualizations...")
    # ANOVA
    f_vals, _ = f_classif(X_hand, y)
    top_idx = np.argsort(np.nan_to_num(f_vals))[-20:]
    plt.figure(figsize=(12, 8)); plt.barh([f"feature_{i}" for i in top_idx], f_vals[top_idx], color='#2ca02c')
    plt.title("Top-20 Feature Separability"); plt.tight_layout(); plt.savefig(out / "feature_separability.png", dpi=200); plt.close()

    # GLCM
    means = pd.DataFrame(all_glcm).groupby('class').mean().reindex(CLASS_NAMES)
    means_norm = pd.DataFrame(MinMaxScaler().fit_transform(means), index=means.index, columns=means.columns)
    plt.figure(figsize=(10, 8)); sns.heatmap(means_norm, annot=True, fmt=".2f", cmap="viridis")
    plt.title("GLCM Texture Heatmap"); plt.tight_layout(); plt.savefig(out / "glcm_heatmap.png", dpi=200); plt.close()

    # t-SNE
    Z = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_deep)
    plt.figure(figsize=(10, 8))
    for i in range(9):
        mask = y == i
        if mask.any(): plt.scatter(Z[mask, 0], Z[mask, 1], label=CLASS_NAMES[i], alpha=0.7, s=20)
    plt.title("t-SNE Embeddings"); plt.legend(bbox_to_anchor=(1.05, 1)); plt.tight_layout(); plt.savefig(out / "cnn_tsne.png", dpi=200); plt.close()

    # RGB
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(9):
        ax = axes[i // 3, i % 3]; profs = [p for p in all_rgb if p['class'] == i]
        if profs:
            ax.plot(np.mean([p['r'] for p in profs], axis=0), color='red')
            ax.plot(np.mean([p['g'] for p in profs], axis=0), color='green')
            ax.plot(np.mean([p['b'] for p in profs], axis=0), color='blue')
        ax.set_title(CLASS_NAMES[i])
    plt.tight_layout(); plt.savefig(out / "rgb_profiles.png", dpi=200); plt.close()

    print(f"\nSUCCESS: Feature report saved in {out}")

if __name__ == "__main__":
    generate_feature_report()
