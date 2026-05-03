import os
import time
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Import project modules
from src.preprocessing import run_full_pipeline, resize_image, reduce_noise, convert_colour_space, apply_clahe, remove_hair, normalise_image
from src.dataset import build_dataframe

def generate_preprocessing_report(config_path="config.yaml", out_dir="outputs/preprocessing_report"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    df = build_dataframe(config)
    
    # 1. Timing Analysis
    print("Performing timing analysis...")
    sample_img_path = df.iloc[0]['image_path']
    img_bgr = cv2.imread(sample_img_path)
    
    timings = {}
    n_runs = 5
    
    for _ in range(n_runs):
        # Resize
        t0 = time.time()
        step1 = resize_image(img_bgr, size=300)
        timings['Resize'] = timings.get('Resize', 0) + (time.time() - t0)
        
        # Noise
        t0 = time.time()
        step2 = reduce_noise(step1)
        timings['Noise'] = timings.get('Noise', 0) + (time.time() - t0)
        
        # Colour
        t0 = time.time()
        step3 = convert_colour_space(step2)
        timings['Colour'] = timings.get('Colour', 0) + (time.time() - t0)
        
        # CLAHE
        t0 = time.time()
        step4 = apply_clahe(step2)
        timings['CLAHE'] = timings.get('CLAHE', 0) + (time.time() - t0)
        
        # Hair Removal
        t0 = time.time()
        step5 = remove_hair(step4)
        timings['HairRemoval'] = timings.get('HairRemoval', 0) + (time.time() - t0)
        
        # Normalise
        t0 = time.time()
        step6 = normalise_image(step5)
        timings['Normalise'] = timings.get('Normalise', 0) + (time.time() - t0)

    avg_timings = {k: (v / n_runs) * 1000 for k, v in timings.items()} # ms
    
    # Plot Timing
    plt.figure(figsize=(12, 5))
    steps = list(avg_timings.keys())
    values = list(avg_timings.values())
    bars = plt.bar(steps, values, color='#e55353', alpha=0.8)
    plt.title("Preprocessing Timing per Step", fontsize=14)
    plt.ylabel("Mean ms / image")
    plt.grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out / "preprocessing_timing.png", dpi=300)
    plt.close()

    # 2. Histogram Shift (3x3)
    print("Generating histogram shift plot...")
    target_classes = ['MEL', 'NV', 'BCC']
    class_map = {0: 'MEL', 1: 'NV', 2: 'BCC'}
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("Brightness Histogram Shift: Raw vs CLAHE vs Final", fontsize=16)
    
    for row_idx, cls_name in enumerate(target_classes):
        cls_df = df[df['label'] == row_idx].sample(1, random_state=42)
        if cls_df.empty: continue
        
        img_path = cls_df.iloc[0]['image_path']
        img_bgr = cv2.imread(img_path)
        img_resized = resize_image(img_bgr, 300)
        
        # Stages
        raw_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        clahe_img = apply_clahe(reduce_noise(img_resized))
        clahe_gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
        final_img = remove_hair(clahe_img)
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        
        stages = [raw_gray, clahe_gray, final_gray]
        titles = ["Raw", "After CLAHE", "After All"]
        
        for col_idx, (stage_gray, title) in enumerate(zip(stages, titles)):
            ax = axes[row_idx, col_idx]
            ax.hist(stage_gray.ravel(), bins=64, range=(0, 256), color='tab:blue', alpha=0.8, histtype='step', linewidth=1.5)
            if row_idx == 0: ax.set_title(title)
            if col_idx == 0: ax.set_ylabel(cls_name, fontweight='bold')
            ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out / "histogram_shift.png", dpi=300)
    plt.close()

    # 3. Preprocessing Pipeline Grid (4x6)
    print("Generating 4x6 pipeline grid...")
    sample_indices = [0, 10, 100, 500] # Just some variety
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle("Preprocessing Pipeline Grid (4x6)", fontsize=16)
    
    col_titles = ["Original", "Denoised", "Colour", "CLAHE", "No Hair", "Normalised"]
    
    for row_idx, sample_idx in enumerate(sample_indices):
        img_path = df.iloc[sample_idx]['image_path']
        img_bgr = cv2.imread(img_path)
        
        res = run_full_pipeline(img_bgr)
        
        # Normalised image needs to be scaled for visualization
        norm_vis = res['step6_normalised']
        norm_vis = (norm_vis - norm_vis.min()) / (norm_vis.max() - norm_vis.min() + 1e-8)
        
        imgs = [
            cv2.cvtColor(res['original'], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(res['step2_denoised'], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(res['step3_lab'], cv2.COLOR_BGR2RGB), # Using LAB as 'Colour' visual
            cv2.cvtColor(res['step4_clahe'], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(res['step5_no_hair'], cv2.COLOR_BGR2RGB),
            (norm_vis * 255).astype(np.uint8)
        ]
        
        for col_idx, (img, title) in enumerate(zip(imgs, col_titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            if row_idx == 0: ax.set_title(title)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out / "pipeline_grid.png", dpi=300)
    plt.close()

    # 4. Hair Detection QA (2-column)
    print("Generating Hair Detection QA Samples...")
    # Find images likely to have hair or just sample some
    hair_samples = df.sample(8, random_state=7) 
    
    fig, axes = plt.subplots(8, 2, figsize=(8, 24))
    plt.subplots_adjust(top=0.96)
    fig.suptitle("Hair Detection QA Samples (2-column, 8 examples)", fontsize=14)
    
    axes[0, 0].set_title("Original (with hair)", fontsize=10)
    axes[0, 1].set_title("After removal + lesion contour", fontsize=10)
    
    for i, (_, row) in enumerate(hair_samples.iterrows()):
        img_path = row['image_path']
        img_bgr = cv2.imread(img_path)
        img_resized = resize_image(img_bgr, 300)
        
        # Step 5 removal
        no_hair = remove_hair(apply_clahe(reduce_noise(img_resized)))
        
        # Dummy contour for visualization (simple threshold on lesion)
        gray = cv2.cvtColor(no_hair, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vis_no_hair = no_hair.copy()
        if contours:
            cv2.drawContours(vis_no_hair, contours, -1, (0, 255, 0), 2)
            
        axes[i, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes[i, 1].imshow(cv2.cvtColor(vis_no_hair, cv2.COLOR_BGR2RGB))
        
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out / "hair_qa.png", dpi=300)
    plt.close()
    
    print(f"\nSUCCESS: Preprocessing report generated in {out}")

if __name__ == "__main__":
    generate_preprocessing_report()
