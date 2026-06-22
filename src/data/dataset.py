"""
dataset.py — PyTorch Dataset for the BN-HTRd line-image/text pairs.

V2 changes from V1:
  - Tokenization uses AutoTokenizer (XLM-RoBERTa) instead of the custom
    BanglaGraphemeTokenizer.  XLM-R's SentencePiece model handles all Bangla
    conjuncts natively without any special-case logic.
  - preprocessing.py now wraps ViTImageProcessor, so pixel values are already
    resized to 224×224 and normalized to the ViT's expected range.
  - The Dataset accepts both absolute image paths (CSV column starting with '/')
    and bare filenames that are resolved against img_dir.
  - Labels use -100 for padding positions so that nn.CrossEntropyLoss /
    VisionEncoderDecoderModel.forward() ignores them automatically.
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional

from src.data.preprocessing import ImageTransform, preprocess_cv, load_image


class LineDataset(Dataset):
    """
    Loads (image, text) pairs from a CSV file produced by prepare_data.py.

    CSV columns expected:
        image : filename (e.g. '129_1_19.jpg') or absolute path
        text  : Bengali ground-truth string
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        tokenizer,
        transform: ImageTransform,
        max_length: int = 128,
    ):
        self.df         = pd.read_csv(csv_file)
        self.img_dir    = img_dir
        self.tokenizer  = tokenizer
        self.transform  = transform
        self.max_length = max_length

        # Validate columns
        required = {"image", "text"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV {csv_file} is missing columns: {missing}")

        # Drop rows with NaN in essential columns
        self.df = self.df.dropna(subset=["image", "text"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Load image ────────────────────────
        img_val = str(row["image"])
        if os.path.isabs(img_val):
            img_path = img_val
        else:
            img_path = os.path.join(self.img_dir, img_val)

        try:
            # Use OpenCV for CLAHE + denoise, then get a PIL RGB image
            cv_img  = load_image(img_path)
            pil_img = preprocess_cv(cv_img)   # returns PIL RGB
        except Exception:
            # Fallback: open directly with PIL (e.g. PNG already clean)
            pil_img = Image.open(img_path).convert("RGB")

        pixel_values = self.transform(pil_img)   # (3, 224, 224) float32

        # ── Tokenize text ─────────────────────
        text    = str(row["text"]).strip()
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Shape: (1, max_length) → squeeze to (max_length,)
        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Build labels: set padding positions to -100 so they are ignored in loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,        # (3, 224, 224)
            "labels":       labels,              # (max_length,)  -100 at pads
            "attention_mask": attention_mask,    # (max_length,)
            "text":         text,                # raw string for metrics
        }


def collate_fn(batch: list) -> dict:
    """Stack a list of dataset items into a single batch dict."""
    pixel_values   = torch.stack([item["pixel_values"]   for item in batch])
    labels         = torch.stack([item["labels"]         for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    texts          = [item["text"] for item in batch]
    return {
        "pixel_values":   pixel_values,    # (B, 3, 224, 224)
        "labels":         labels,          # (B, max_length)
        "attention_mask": attention_mask,  # (B, max_length)
        "texts":          texts,           # list[str]
    }
