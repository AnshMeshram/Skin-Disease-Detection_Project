from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert_bgr_u8(arr: np.ndarray) -> None:
    assert isinstance(arr, np.ndarray), "Input must be numpy array"
    assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
    assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected (H,W,3), got {arr.shape}"


def _assert_no_nan(arr: np.ndarray) -> None:
    assert not np.any(np.isnan(arr.astype(float))), "NaN in output"


def _to_bgr_uint8(img_input: Any) -> np.ndarray:
    if isinstance(img_input, np.ndarray):
        _assert_bgr_u8(img_input)
        return img_input
    if isinstance(img_input, Image.Image):
        rgb = np.array(img_input.convert("RGB"))
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _assert_bgr_u8(out)
        return out
    raise TypeError("Input must be numpy BGR uint8 or PIL image")


def resize_image(img_bgr: np.ndarray, size: int = 300) -> np.ndarray:
    img_bgr = _to_bgr_uint8(img_bgr)
    original_shape = img_bgr.shape

    interpolation = cv2.INTER_AREA
    if img_bgr.shape[0] < size or img_bgr.shape[1] < size:
        interpolation = cv2.INTER_CUBIC

    out = cv2.resize(img_bgr, (size, size), interpolation=interpolation)
    assert out.shape == (size, size, 3)
    assert out.dtype == np.uint8
    _assert_no_nan(out)
    print(f"Resize: {original_shape} -> {out.shape}")
    return out


def reduce_noise(
    img_bgr: np.ndarray,
    median_ksize: int = 3,
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 0.8,
) -> np.ndarray:
    img_bgr = _to_bgr_uint8(img_bgr)
    assert median_ksize % 2 == 1 and gaussian_ksize % 2 == 1, "Kernel sizes must be odd"
    assert gaussian_sigma <= 1.0, "Sigma must be <= 1.0 for dermoscopy"

    out = cv2.medianBlur(img_bgr, median_ksize)
    out = cv2.GaussianBlur(out, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

    assert out.shape == img_bgr.shape
    assert out.dtype == np.uint8
    _assert_no_nan(out)
    return out


def convert_colour_space(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    img_bgr = _to_bgr_uint8(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    assert lab.shape == img_bgr.shape
    assert hsv.shape == img_bgr.shape
    assert lab.dtype == np.uint8 and hsv.dtype == np.uint8
    _assert_no_nan(lab)
    _assert_no_nan(hsv)
    return {"lab": lab, "hsv": hsv, "bgr": img_bgr}


def apply_clahe(
    img_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    img_bgr = _to_bgr_uint8(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_chan)
    out = cv2.cvtColor(cv2.merge((l_clahe, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"
    assert out.shape == img_bgr.shape
    _assert_no_nan(out)
    print(f"CLAHE applied: clip_limit={clip_limit}, tile={tile_grid_size}")
    return out


def remove_hair(img_bgr: np.ndarray, kernel_size: int = 17, inpaint_radius: int = 3) -> np.ndarray:
    img_bgr = _to_bgr_uint8(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)[1]
    out = cv2.inpaint(img_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)

    assert out.shape == img_bgr.shape
    assert out.dtype == np.uint8
    _assert_no_nan(out)
    print(f"Hair removal: {int(hair_mask.sum() // 255)} hair pixels inpainted")
    return out


def normalise_image(img_bgr: np.ndarray) -> np.ndarray:
    img_bgr = _to_bgr_uint8(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixel_normalised = img_rgb.astype(np.float32) / 255.0
    imagenet_normalised = (pixel_normalised - MEAN) / STD
    assert imagenet_normalised.dtype == np.float32
    assert imagenet_normalised.shape == (300, 300, 3)
    assert not np.any(np.isnan(imagenet_normalised)), "NaN in output"
    return imagenet_normalised


def to_tensor_gpu(normalised_np: np.ndarray, device: str = "cuda") -> torch.Tensor:
    assert isinstance(normalised_np, np.ndarray), "Input must be numpy array"
    assert normalised_np.dtype == np.float32, f"Expected float32, got {normalised_np.dtype}"
    assert normalised_np.ndim == 3 and normalised_np.shape[2] == 3, f"Expected (H,W,3), got {normalised_np.shape}"

    tensor = torch.from_numpy(normalised_np.transpose(2, 0, 1)).float().unsqueeze(0)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return tensor.to(device)


def _save_intermediate(path: Path, img: np.ndarray) -> None:
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
        return
    cv2.imwrite(str(path), img)


def run_full_pipeline(
    img_input: str | np.ndarray | Image.Image,
    config: dict | None = None,
    save_intermediates: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Runs all 6 steps in exact order and returns all stage artifacts."""
    config = config or {}
    processing_log: list[str] = []

    if isinstance(img_input, str):
        img_path = Path(img_input)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
    else:
        img_bgr = _to_bgr_uint8(img_input)

    step1 = resize_image(img_bgr, size=int(config.get("resize", 300)))
    processing_log.append("Step 1: resize_image")

    step2 = reduce_noise(
        step1,
        median_ksize=int(config.get("median_ksize", 3)),
        gaussian_ksize=int(config.get("gaussian_ksize", 3)),
        gaussian_sigma=float(config.get("gaussian_sigma", 0.8)),
    )
    processing_log.append("Step 2: reduce_noise")

    step3 = convert_colour_space(step2)
    processing_log.append("Step 3: convert_colour_space")

    step4 = apply_clahe(
        step2,
        clip_limit=float(config.get("clip_limit", 2.0)),
        tile_grid_size=tuple(config.get("tile_grid_size", (8, 8))),
    )
    processing_log.append("Step 4: apply_clahe")

    step5 = remove_hair(
        step4,
        kernel_size=int(config.get("hair_kernel_size", 17)),
        inpaint_radius=int(config.get("inpaint_radius", 3)),
    )
    processing_log.append("Step 5: remove_hair")

    step6 = normalise_image(step5)
    step6_tensor = to_tensor_gpu(step6, device=str(config.get("device", "cuda")))
    processing_log.append("Step 6: normalise_image")

    if save_intermediates:
        if out_dir is None:
            raise ValueError("out_dir is required when save_intermediates=True")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _save_intermediate(out_path / "step0_original.png", resize_image(img_bgr, size=300))
        _save_intermediate(out_path / "step1_resized.png", step1)
        _save_intermediate(out_path / "step2_denoised.png", step2)
        _save_intermediate(out_path / "step3_lab.png", step3["lab"])
        _save_intermediate(out_path / "step3_hsv.png", step3["hsv"])
        _save_intermediate(out_path / "step4_clahe.png", step4)
        _save_intermediate(out_path / "step5_no_hair.png", step5)

    return {
        "original": resize_image(img_bgr, size=300),
        "step1_resized": step1,
        "step2_denoised": step2,
        "step3_lab": step3["lab"],
        "step3_hsv": step3["hsv"],
        "step4_clahe": step4,
        "step5_no_hair": step5,
        "step6_normalised": step6,
        "step6_tensor": step6_tensor,
        "processing_log": processing_log,
    }


def preprocess_image(img_bgr: np.ndarray, apply_hair: bool = True, apply_clahe: bool = True) -> np.ndarray:
    _assert_bgr_u8(img_bgr)
    out = resize_image(img_bgr, size=300)
    out = reduce_noise(out)
    if apply_clahe:
        out = globals()["apply_clahe"](out)
    if apply_hair:
        out = remove_hair(out)
    assert out.shape[2] == 3, "Must be 3-channel"
    assert out.dtype == np.uint8, "Must be uint8"
    _assert_no_nan(out)
    return out


def preprocess_pil(pil_image: Image.Image, config: dict | None = None) -> Image.Image:
    """Returns PIL image after steps 1-5 (not normalised)."""
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = run_full_pipeline(img_bgr, config=config)
    processed_bgr = result["step5_no_hair"]
    return Image.fromarray(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))


def save_pipeline_comparison(img_path: str, out_path: str, config: dict | None = None) -> None:
    result = run_full_pipeline(img_path, config=config)
    stages = [
        ("Original", cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB)),
        ("Resize", cv2.cvtColor(result["step1_resized"], cv2.COLOR_BGR2RGB)),
        ("Denoise", cv2.cvtColor(result["step2_denoised"], cv2.COLOR_BGR2RGB)),
        ("CLAHE", cv2.cvtColor(result["step4_clahe"], cv2.COLOR_BGR2RGB)),
        ("HairRem", cv2.cvtColor(result["step5_no_hair"], cv2.COLOR_BGR2RGB)),
        ("Overlay", cv2.addWeighted(cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB), 0.5, cv2.cvtColor(result["step5_no_hair"], cv2.COLOR_BGR2RGB), 0.5, 0)),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    for i, (title, img) in enumerate(stages):
        axes[0, i].imshow(img)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")

        gray = cv2.cvtColor((img * 255).astype(np.uint8) if img.dtype != np.uint8 else img, cv2.COLOR_RGB2GRAY)
        axes[1, i].hist(gray.ravel(), bins=64, color="tab:blue", alpha=0.8)
        axes[1, i].set_title(f"Hist: {title}")

    fig.tight_layout()
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "raw/ISIC_2019_Training_Input/ISIC_0000000.jpg"
    result = run_full_pipeline(img_path, save_intermediates=True, out_dir="outputs/preprocessing_test")
    save_pipeline_comparison(img_path, "outputs/preprocessing_test/comparison_grid.png")
    print("All 6 preprocessing steps completed successfully")
    print(f"Output tensor shape: {result['step6_tensor'].shape}")
    print(f"Log: {chr(10).join(result['processing_log'])}")
