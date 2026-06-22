"""
preprocessing.py — Image preprocessing for the ViT + XLM-RoBERTa V2 pipeline.

Key design choices vs V1:
  - Encoder is now google/vit-base-patch16-384 (Suggestion 1).
    ViTImageProcessor handles the canonical 384×384 resize and normalization.
  - OpenCV CLAHE + fast denoise run BEFORE the processor for clean ink strokes.
  - Training augmentation uses albumentations (Suggestion 7):
      • ElasticTransform — simulates natural pen/paper deformation
      • GridDistortion   — simulates scanner/paper warp
      • Mild photometric: brightness/contrast, Gaussian blur
    These are followed by a lightweight torchvision affine/rotation stage that
    operates on the PIL image (which albumentations returns as a numpy array).
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False

# ──────────────────────────────────────────────
# Low-level OpenCV helpers
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load an image from disk via OpenCV (BGR uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def clahe_equalization(img: np.ndarray,
                       clip_limit: float = 2.0,
                       grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def fast_denoise(img: np.ndarray, h: float = 8.0) -> np.ndarray:
    """Non-local means denoising; reduces ink bleed and scanner noise."""
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def preprocess_cv(img: np.ndarray) -> Image.Image:
    """
    Full OpenCV pre-processing pipeline.
    Returns a PIL RGB image ready for ViTImageProcessor.
    """
    gray = to_grayscale(img)
    equalized = clahe_equalization(gray)
    denoised = fast_denoise(equalized)
    # Convert grayscale → RGB (ViT expects 3-channel input)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# ──────────────────────────────────────────────
# Augmentation pipelines (Suggestion 7: albumentations elastic/grid)
# ──────────────────────────────────────────────

# Stage A: albumentations — operates on uint8 numpy RGB arrays.
# These distortions cannot be replicated with torchvision transforms.
_ALBU_AUGMENT: "A.Compose | None" = None
if _ALBUMENTATIONS_AVAILABLE:
    _ALBU_AUGMENT = A.Compose([
        # ElasticTransform: simulates natural pen-stroke deformation
        A.ElasticTransform(
            alpha=60,
            sigma=8,
            p=0.45,
        ),
        # GridDistortion: simulates paper warp / scanner bed distortion
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.15,
            p=0.35,
        ),
        # Morphological: dilation/erosion mimics thick/thin pen styles
        A.Morphological(
            scale=(1, 2),
            operation="dilation",
            p=0.20,
        ),
        # Photometric (mild — Bangla ink can vary widely in density)
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.55,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.30),
    ])

# Stage B: torchvision — operates on PIL images (RGB).
# Handles geometric transforms that need PIL's fill/interpolation support.
_TV_AUGMENT = transforms.Compose([
    transforms.RandomRotation(degrees=4, fill=255),
    transforms.RandomAffine(
        degrees=3,
        translate=(0.02, 0.02),
        scale=(0.93, 1.07),
        shear=6,
        fill=255,
    ),
    transforms.RandomPerspective(distortion_scale=0.08, p=0.30, fill=255),
])


def _apply_augmentations(img: Image.Image) -> Image.Image:
    """
    Apply the full two-stage augmentation pipeline to a PIL RGB image.

    Stage A: albumentations on uint8 numpy (elastic, grid distortion, morphology).
    Stage B: torchvision on PIL (rotation, affine, perspective).

    Falls back to Stage B only if albumentations is not installed.
    """
    if _ALBUMENTATIONS_AVAILABLE and _ALBU_AUGMENT is not None:
        arr = np.array(img)                       # PIL → uint8 numpy RGB
        augmented = _ALBU_AUGMENT(image=arr)      # albumentations pipeline
        img = Image.fromarray(augmented["image"]) # back to PIL
    img = _TV_AUGMENT(img)                        # torchvision stage
    return img


class ImageTransform:
    """
    Wraps ViTImageProcessor with a two-stage augmentation pipeline.

    Stage A (albumentations): ElasticTransform, GridDistortion, morphological ops.
    Stage B (torchvision):    Rotation, affine, perspective.
    ViTImageProcessor then resizes to 384×384 and normalizes for ViT-384.

    Args:
        processor: ViTImageProcessor from 'google/vit-base-patch16-384'.
        augment:   If True, apply training-time augmentations.
    """

    def __init__(self, processor: ViTImageProcessor, augment: bool = False):
        self.processor = processor
        self.augment   = augment

        if augment and not _ALBUMENTATIONS_AVAILABLE:
            print(
                "⚠️  albumentations not found — falling back to torchvision-only "
                "augmentation. Run: pip install albumentations"
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Args:
            img: PIL.Image (RGB)
        Returns:
            pixel_values: float32 tensor of shape (3, 384, 384)
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.augment:
            img = _apply_augmentations(img)

        # ViTImageProcessor: resize to 384×384, normalize to ViT's expected range
        encoding = self.processor(images=img, return_tensors="pt")
        # squeeze batch dim → (3, 384, 384)
        return encoding["pixel_values"].squeeze(0)
