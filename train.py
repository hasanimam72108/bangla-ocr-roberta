"""
train.py — Entry-point for Bangla OCR V2 training.

Usage (local):
    python train.py --config configs/train_config.yaml

Usage (Kaggle):
    See kaggle_cells/03_train.py for the paste-ready version.
"""

import os
import sys
import yaml
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.data.preprocessing import ImageTransform
from src.data.dataset import LineDataset, collate_fn
from src.models.ocr_model import build_model, count_parameters
from src.training.trainer import Trainer


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bangla OCR V2 — Training")
    parser.add_argument("--config",      type=str, default="configs/train_config.yaml")
    parser.add_argument("--output_dir",  type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to last_model.pt to resume training")
    parser.add_argument("--accumulate",  type=int, default=1,
                        help="Gradient accumulation steps (simulate larger batch)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()

    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ────────────────────────────
    print("\n--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["decoder_name"])
    print(f"Tokenizer vocab size : {tokenizer.vocab_size}")
    print(f"BOS / EOS token ids  : {tokenizer.cls_token_id} / {tokenizer.sep_token_id}")

    # ── Image processor ───────────────────────
    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained(cfg["model"]["encoder_name"])

    train_transform = ImageTransform(processor, augment=cfg["data"]["augment"])
    val_transform   = ImageTransform(processor, augment=False)

    # ── Datasets ─────────────────────────────
    print("\n--- Loading datasets ---")
    train_ds = LineDataset(
        csv_file   = cfg["data"]["train_csv"],
        img_dir    = cfg["data"]["train_img_dir"],
        tokenizer  = tokenizer,
        transform  = train_transform,
        max_length = cfg["model"]["max_length"],
    )
    val_ds = LineDataset(
        csv_file   = cfg["data"]["val_csv"],
        img_dir    = cfg["data"]["val_img_dir"],
        tokenizer  = tokenizer,
        transform  = val_transform,
        max_length = cfg["model"]["max_length"],
    )

    num_workers = cfg["data"].get("num_workers", 2)
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"] * 2,
        shuffle     = False,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )
    print(f"Train : {len(train_ds)} samples   Val : {len(val_ds)} samples")

    # ── Model ────────────────────────────────
    print("\n--- Building model ---")
    model = build_model(
        encoder_name  = cfg["model"]["encoder_name"],
        decoder_name  = cfg["model"]["decoder_name"],
        tokenizer     = tokenizer,
        max_length    = cfg["model"]["max_length"],
        dropout       = cfg["model"]["dropout"],
        freeze_encoder= True,   # always start frozen; Trainer handles un-freeze
    )
    model.to(device)

    counts = count_parameters(model)
    print(f"Encoder params   : {counts['encoder']:,}")
    print(f"Decoder params   : {counts['decoder']:,}")
    print(f"Total params     : {counts['total']:,}")
    print(f"Trainable params : {counts['trainable']:,}")

    # ── Optional resume ───────────────────────
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  → Continuing from epoch {start_epoch}")

    # ── Train ─────────────────────────────────
    print("\n--- Starting training ---")
    trainer = Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        tokenizer        = tokenizer,
        config           = cfg,
        device           = device,
        output_dir       = args.output_dir,
        accumulate_steps = args.accumulate,
    )
    best_cer = trainer.run()

    print(f"\nTraining complete! Best validation CER : {best_cer:.4f}")
    print(f"Best model saved to : {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
