""""
scripts/augment_images.py
--------------------------
Synthetic image augmentation for Flickr30k / COCO images.
Generates augmented copies saved to data/images/augmented/
and updates the captions JSON to include augmented entries.

Techniques:
  1. Horizontal flip
  2. Random crop + resize
  3. Color jitter (brightness, contrast, saturation)
  4. Gaussian blur + noise
  5. Rotation (±15°)

Usage:
    python scripts/augment_images.py --input data/processed/flickr30k_captions.json
                                      --output data/processed/flickr30k_augmented.json
                                      --methods flip crop color
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_IMAGES  = ROOT / "data" / "images" / "augmented"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_IMAGES.mkdir(parents=True, exist_ok=True)


# ── Individual augmentation functions ─────────────────────────────────────────

def aug_flip(img: Image.Image) -> Image.Image:
    """Horizontal mirror flip."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def aug_crop(img: Image.Image, crop_pct: float = 0.85) -> Image.Image:
    """Random crop keeping crop_pct of the image, then resize back."""
    w, h = img.size
    new_w = int(w * crop_pct)
    new_h = int(h * crop_pct)
    left = random.randint(0, w - new_w)
    top  = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)


def aug_color(img: Image.Image) -> Image.Image:
    """Random brightness, contrast, and saturation jitter."""
    # Brightness: 0.7–1.3
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    # Contrast: 0.8–1.2
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    # Color saturation: 0.8–1.2
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img


def aug_blur(img: Image.Image) -> Image.Image:
    """Gaussian blur with a small random radius."""
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def aug_noise(img: Image.Image, sigma: float = 10.0) -> Image.Image:
    """Add Gaussian noise to pixel values."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def aug_rotate(img: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """Random rotation ±max_angle degrees, white fill for corners."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))


# ── Method dispatch ────────────────────────────────────────────────────────────

AUG_FUNCS = {
    "flip":   aug_flip,
    "crop":   aug_crop,
    "color":  aug_color,
    "blur":   aug_blur,
    "noise":  aug_noise,
    "rotate": aug_rotate,
}


def augment_image(img: Image.Image, method: str) -> Image.Image:
    fn = AUG_FUNCS.get(method)
    if fn is None:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(AUG_FUNCS)}")
    return fn(img)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_augmentation(
    input_path: str,
    output_path: str,
    methods: List[str],
    max_images: int = None,
    quality: int = 85,
):
    """
    Load image records, apply each augmentation method, save new images + updated JSON.
    """
    logger.info(f"Loading image records from {input_path}...")
    with open(input_path) as f:
        original_records = json.load(f)

    if max_images:
        original_records = original_records[:max_images]

    logger.info(f"Loaded {len(original_records)} image records.")
    all_records = list(original_records)

    for method in methods:
        if method not in AUG_FUNCS:
            logger.warning(f"Unknown method '{method}'. Skipping.")
            continue

        logger.info(f"Applying '{method}' augmentation...")
        aug_records = []

        for rec in tqdm(original_records, desc=f"Image aug [{method}]"):
            src_path = Path(rec["image_path"])
            if not src_path.exists():
                continue

            try:
                img = Image.open(src_path).convert("RGB")
                aug_img = augment_image(img, method)

                # Save augmented image
                new_name = f"{src_path.stem}_aug_{method}.jpg"
                save_path = DATA_IMAGES / new_name
                aug_img.save(save_path, "JPEG", quality=quality)

                aug_records.append({
                    "image_id":   f"{rec['image_id']}_aug_{method}",
                    "image_path": str(save_path),
                    "caption":    rec.get("caption", ""),
                    "source":     "augmented",
                    "method":     method,
                    "orig_id":    rec["image_id"],
                })
            except Exception as e:
                logger.warning(f"Failed on {src_path.name}: {e}")

        all_records.extend(aug_records)
        logger.info(f"Added {len(aug_records)} images via '{method}'. Total: {len(all_records)}")

    logger.info(f"Saving {len(all_records)} records to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(all_records, f)
    logger.info("Done.")
    logger.info(f"Augmentation ratio: {len(all_records)/len(original_records):.1f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment image dataset")
    parser.add_argument("--input",  default=str(DATA_PROCESSED / "flickr30k_captions.json"))
    parser.add_argument("--output", default=str(DATA_PROCESSED / "flickr30k_augmented.json"))
    parser.add_argument("--methods", nargs="+",
                        default=["flip", "crop", "color"],
                        choices=list(AUG_FUNCS),
                        help="Augmentation methods to apply")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    run_augmentation(
        input_path=args.input,
        output_path=args.output,
        methods=args.methods,
        max_images=args.max_images,
    )