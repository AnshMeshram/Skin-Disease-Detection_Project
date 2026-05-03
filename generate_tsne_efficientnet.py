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
from src.dataset import build_dataframe, create_holdout_dataloaders, CLASS_NAMES
from src.model_factory import build_model

def generate_tsne_efficientnet(config_path="config.yaml", samples=1000):
    print(f"--- SkinGuard t-SNE Generator (EfficientNet-B3 | Test Set) ---")
    
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force model type for this script
    model_name = "efficientnet_b3"
    checkpoint_path = f"outputs/{model_name}/fold_0/best.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data (Holdout Split)
    print("Building holdout split dataloaders...")
    df = build_dataframe(config)
    _, _, test_loader, _ = create_holdout_dataloaders(df, config)
    
    print(f"Using Test Set with {len(test_loader.dataset)} samples.")

    # 3. Load Model
    print(f"Loading {model_name} from {checkpoint_path}...")
    model = build_model(model_name, num_classes=9, pretrained=False)
    state = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state' in state:
        state = state['model_state']
    elif 'state_dict' in state:
        state = state['state_dict']
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 4. Extract Features
    # We want features from before the final head
    print("Extracting features from Test Set...")
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader):
            imgs = imgs.to(device)
            # Pass through backbone and attention, but skip the head
            x = model.backbone(imgs)
            x = model.sa(x)
            x = model.pool(x).flatten(1)
            
            features.append(x.cpu().numpy())
            labels.append(lbls.numpy())

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    
    print(f"Extracted {len(X)} feature vectors of dim {X.shape[1]}")

    # 5. Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 6. Plot
    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    
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
        if mask.any():
            plt.scatter(
                X_embedded[mask, 0], 
                X_embedded[mask, 1], 
                label=CLASS_NAMES[i],
                alpha=0.8,
                s=50,
                edgecolors='white',
                linewidths=0.5,
                color=colors[i]
            )

    plt.title(f"t-SNE Visualization: EfficientNet-B3 (Test Set Clusters)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_path = "tsne_plot_efficientnet_test.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSUCCESS: EfficientNet t-SNE plot saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_tsne_efficientnet()
