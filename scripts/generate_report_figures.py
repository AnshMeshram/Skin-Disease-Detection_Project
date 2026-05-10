from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "Healthy"]


def _pick_gradcam_file(class_idx: int) -> Path:
    base = Path("outputs") / "gradcam"
    candidates = [
        base / f"efficientnet_b3_class{class_idx}_1.png",
        base / f"efficientnet_b3_class{class_idx}_2.png",
        base / f"efficientnet_b3_class{class_idx}_3.png",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Missing Grad-CAM image for class {class_idx}: {candidates}")


def _crop_original_overlay(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    third = w // 3
    if third <= 0:
        raise ValueError("Invalid Grad-CAM image width")
    original = img.crop((0, 0, third, h))
    overlay = img.crop((2 * third, 0, 3 * third, h))
    return original, overlay


def generate_gradcam_grid(out_path: Path) -> None:
    target = 240
    gap = 10
    label_h = 24
    grid_gap = 18

    pair_w = target * 2 + gap
    pair_h = target
    cell_w = pair_w
    cell_h = label_h + pair_h

    cols = 3
    rows = 3
    img_w = cols * cell_w + (cols - 1) * grid_gap
    img_h = rows * cell_h + (rows - 1) * grid_gap

    canvas = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, name in enumerate(CLASS_NAMES):
        src = _pick_gradcam_file(idx)
        raw = Image.open(src).convert("RGB")
        original, overlay = _crop_original_overlay(raw)
        original = original.resize((target, target), Image.LANCZOS)
        overlay = overlay.resize((target, target), Image.LANCZOS)

        row = idx // cols
        col = idx % cols
        x0 = col * (cell_w + grid_gap)
        y0 = row * (cell_h + grid_gap)

        # Label
        label = name
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x0 + (cell_w - text_w) // 2
        draw.text((text_x, y0), label, fill=(20, 20, 20), font=font)

        # Pair images (original left, overlay right)
        img_y = y0 + label_h
        canvas.paste(original, (x0, img_y))
        canvas.paste(overlay, (x0 + target + gap, img_y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def generate_preprocessing_pipeline(out_path: Path) -> None:
    src = Path("outputs") / "analysis" / "preprocessing" / "pipeline_grid.png"
    if not src.exists():
        raise FileNotFoundError(f"Pipeline grid not found: {src}")
    img = Image.open(src).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    generate_gradcam_grid(Path("outputs") / "gradcam" / "efficientnet b3 gradcam grid.png")
    generate_preprocessing_pipeline(Path("outputs") / "plots" / "preprocessing samples.png")
    print("Saved: outputs/gradcam/efficientnet b3 gradcam grid.png")
    print("Saved: outputs/plots/preprocessing samples.png")


if __name__ == "__main__":
    main()
