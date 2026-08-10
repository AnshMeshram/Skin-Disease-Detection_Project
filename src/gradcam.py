from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.dataset import build_dataframe, create_fold_dataloaders
from src.model_factory import build_model


def _resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    parts = path.split(".")
    mod = root
    for p in parts:
        if p.isdigit():
            mod = mod[int(p)]
        else:
            mod = getattr(mod, p)
    return mod


def get_target_layer(model_name: str, model) -> str:
    model_name = model_name.lower()
    if "efficientnet" in model_name:
        # Check if it's our SkinEfficientNetB3 wrapper
        if hasattr(model, "backbone"):
            trunk = model.backbone
            if hasattr(trunk, "blocks"): # Timm variant
                return "backbone.blocks"
            if hasattr(trunk, "features"): # Torchvision variant
                return "backbone.features"
        return "backbone"
    if "inception" in model_name:
        return "backbone.Mixed_7c"
    if "convnext" in model_name:
        return "backbone.features.7"
    return "backbone"


class GradCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None

        target_layer = _resolve_module(self.model, target_layer_name)

        def _fw_hook(_module, _inp, out):
            self.activations = out.detach()

        def _bw_hook(_module, grad_input, grad_output):
            # grad_output is tuple with gradient wrt layer output.
            self.gradients = grad_output[0].detach()

        self._fw = target_layer.register_forward_hook(_fw_hook)
        self._bw = target_layer.register_full_backward_hook(_bw_hook)

    def remove_hooks(self):
        if self._fw is not None:
            self._fw.remove()
            self._fw = None
        if self._bw is not None:
            self._bw.remove()
            self._bw = None

    def generate(self, input_tensor, class_idx=None) -> np.ndarray:
        input_tensor.requires_grad = True
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        # grads, activations: (B,C,H,W)
        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        h, w = int(input_tensor.shape[2]), int(input_tensor.shape[3])
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cam = np.clip(cam, 0.0, 1.0).astype(np.float32)
        return cam

    def overlay(self, img_np, heatmap, alpha=0.4) -> np.ndarray:
        img = np.asarray(img_np)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        heat_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img, 1.0 - alpha, heat_rgb, alpha, 0)
        return overlay.astype(np.uint8)


def _tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().transpose(1, 2, 0)
    # Inverse ImageNet normalization (dataset uses mean/std normalize).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = arr * std + mean
    arr = np.clip(arr, 0, 1)
    return (arr * 255.0).astype(np.uint8)


def generate_gradcam_grid(model, dataloader, class_names, out_dir, n_per_class=3, model_name="model"):
    device = next(model.parameters()).device
    target_layer = get_target_layer(model_name, model)
    cam = GradCAM(model, target_layer)

    collected = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)
            preds = logits.argmax(dim=1)

            for i in range(inputs.size(0)):
                true_cls = int(labels[i].item())
                pred_cls = int(preds[i].item())
                if pred_cls != true_cls:
                    continue
                if len(collected[true_cls]) >= n_per_class:
                    continue
                collected[true_cls].append((inputs[i].detach().cpu(), true_cls, pred_cls))

            done = all(len(collected[c]) >= n_per_class for c in range(len(class_names)))
            if done:
                break

    out_root = Path(out_dir) / "gradcam"
    out_root.mkdir(parents=True, exist_ok=True)

    # Save per-sample 3-panel images.
    for cls_idx in range(len(class_names)):
        for j, (inp_cpu, true_cls, pred_cls) in enumerate(collected.get(cls_idx, []), start=1):
            x = inp_cpu.unsqueeze(0).to(device)
            heat = cam.generate(x, class_idx=pred_cls)
            img_rgb = _tensor_to_rgb_uint8(inp_cpu)
            ov = cam.overlay(img_rgb, heat, alpha=0.4)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original")
            axes[1].imshow(heat, cmap="jet")
            axes[1].set_title("Heatmap")
            axes[2].imshow(ov)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")

            img_out = out_root / f"{model_name}_class{cls_idx}_{j}.png"
            fig.tight_layout()
            fig.savefig(img_out, dpi=250)
            plt.close(fig)
            print(f"Saved GradCAM sample: {img_out}")

    # Save class grid: 9 rows x n_per_class cols (overlay thumbnails).
    rows = len(class_names)
    cols = n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.7 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r in range(rows):
        samples = collected.get(r, [])
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(samples):
                inp_cpu, true_cls, pred_cls = samples[c]
                x = inp_cpu.unsqueeze(0).to(device)
                heat = cam.generate(x, class_idx=pred_cls)
                img_rgb = _tensor_to_rgb_uint8(inp_cpu)
                ov = cam.overlay(img_rgb, heat, alpha=0.4)
                ax.imshow(ov)
            if c == 0:
                ax.set_ylabel(class_names[r], rotation=0, labelpad=30, va="center")

    grid_out = out_root / f"{model_name}_gradcam_grid.png"
    fig.tight_layout()
    fig.savefig(grid_out, dpi=300)
    plt.close(fig)
    print(f"Saved GradCAM grid: {grid_out}")

    cam.remove_hooks()
    return grid_out


if __name__ == "__main__":
    cfg_path = Path("config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = str(config.get("model", "efficientnet_b3"))
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]

    df = build_dataframe(config, prefer_preprocessed=bool(config.get("preprocessing", {}).get("use_preprocessed", False)))
    _tr_dl, val_dl, _w = create_fold_dataloaders(df, config, fold_idx=0, preprocess_fn=None)

    model = build_model(model_name, num_classes=9, pretrained=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ckpt_path = Path("outputs") / model_name / "fold_0" / "best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)

    out = generate_gradcam_grid(
        model=model,
        dataloader=val_dl,
        class_names=class_names,
        out_dir="outputs",
        n_per_class=3,
        model_name=model_name,
    )
    print("GradCAM saved to outputs/gradcam/")
    print(f"Grid: {out}")
