import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# Import project modules
from src.dataset import build_dataframe, SkinLesionDataset, get_albumentations_val, CLASS_NAMES
from src.inception_v3 import SkinInceptionV3

def generate_tsne_plot(config_path="config.yaml", checkpoint_path="outputs/inception_v3/fold_0/best.pth", samples=800):
    print(f"--- SkinGuard t-SNE Generator ---")
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading dataset...")
    df = build_dataframe(config)
    # Sample subset for faster t-SNE
    df_sample = df.groupby('label').sample(n=min(samples // 9, len(df)//9), random_state=42, replace=False)
    
    val_tf = get_albumentations_val(config)
    ds = SkinLesionDataset(df_sample, transform=val_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    # 3. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = SkinInceptionV3(num_classes=9, pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle state_dict if it was saved as a whole dict (common variants)
    if 'model_state' in state:
        state = state['model_state']
    elif 'state_dict' in state:
        state = state['state_dict']
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 4. Extract Features
    print("Extracting features...")
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to(device)
            # Get bottleneck features from backbone
            f = model.backbone(imgs)
            if isinstance(f, tuple):
                f = f[0]
            features.append(f.cpu().numpy())
            labels.append(lbls.numpy())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    
    print(f"Extracted {len(X)} feature vectors of dim {X.shape[1]}")

    # 5. Run t-SNE
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 6. Plot
    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    
    # Define clinical colors for classes
    colors = [
        '#e11d48', # MEL (Red)
        '#2563eb', # NV (Blue)
        '#d97706', # BCC (Amber)
        '#7c3aed', # AK (Purple)
        '#059669', # BKL (Green)
        '#db2777', # DF (Pink)
        '#4b5563', # VASC (Gray)
        '#ea580c', # SCC (Orange)
        '#10b981'  # Healthy (Emerald)
    ]

    for i in range(9):
        mask = (y == i)
        plt.scatter(
            X_embedded[mask, 0], 
            X_embedded[mask, 1], 
            label=CLASS_NAMES[i],
            alpha=0.7,
            s=40,
            edgecolors='white',
            linewidths=0.5,
            color=colors[i]
        )

    plt.title("t-SNE Visualization of Skin Lesion Features (InceptionV3 Bottleneck)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save to root as requested
    output_path = "tsne_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS: t-SNE plot saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_tsne_plot()
