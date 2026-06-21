"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE CELL 03 — Training                                           ║
║  Paste this entire script into a Kaggle code cell and run it.        ║
╚══════════════════════════════════════════════════════════════════════╝

This cell:
  1. Loads the XLM-RoBERTa tokenizer and ViT image processor.
  2. Builds the train/val datasets.
  3. Instantiates the ViT + XLM-RoBERTa model.
  4. Runs the Trainer (with phased encoder un-freezing).

Outputs are written to /kaggle/working/checkpoints/:
  last_model.pt  — always overwritten (disk-safe)
  best_model.pt  — only written when CER improves
"""

import os, sys

# ── Make sure project root is on path ────────────────────────────────
PROJ = "/kaggle/working/bangla-ocr-roberta"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
import yaml
from transformers import AutoTokenizer, ViTImageProcessor
from torch.utils.data import DataLoader

from src.data.preprocessing import ImageTransform
from src.data.dataset        import LineDataset, collate_fn
from src.models.ocr_model    import build_model, count_parameters
from src.training.trainer    import Trainer

# ─────────────────────────────────────────────────────────────────────
# ✏️  Configuration (mirrors configs/train_config.yaml)
# ─────────────────────────────────────────────────────────────────────
CFG = {
    "model": {
        "encoder_name":          "google/vit-base-patch16-224",
        "decoder_name":          "xlm-roberta-base",
        "image_size":            224,
        "max_length":            128,
        "dropout":               0.1,
        "label_smoothing":       0.1,
        "freeze_encoder_epochs": 5,
    },
    "training": {
        "batch_size":          16,
        "epochs":              80,
        "learning_rate":       3e-5,
        "encoder_lr":          5e-6,
        "weight_decay":        0.01,
        "warmup_steps":        300,
        "gradient_clip":       1.0,
        "mixed_precision":     True,
        "early_stop_patience": 15,
        "save_every":          1,
        "log_every":           50,
        "num_beams":           4,
    },
    "data": {
        "train_csv":     "/kaggle/working/data/train.csv",
        "val_csv":       "/kaggle/working/data/val.csv",
        "train_img_dir": "/kaggle/working/data/train",
        "val_img_dir":   "/kaggle/working/data/train",
        "augment":       True,
        "num_workers":   2,
    },
}

# Gradient accumulation: set > 1 to simulate a larger effective batch
# e.g. ACCUM=2 with batch_size=16 → effective batch of 32
ACCUM_STEPS = 2
OUTPUT_DIR  = "/kaggle/working/checkpoints"
RESUME_FROM = None  # set to "/kaggle/working/checkpoints/last_model.pt" to resume

# ─────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────
# Tokenizer & Image Processor
# ─────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading tokenizer & processor...")
tokenizer = AutoTokenizer.from_pretrained(CFG["model"]["decoder_name"])
processor = ViTImageProcessor.from_pretrained(CFG["model"]["encoder_name"])

train_transform = ImageTransform(processor, augment=CFG["data"]["augment"])
val_transform   = ImageTransform(processor, augment=False)

# ─────────────────────────────────────────────────────────────────────
# Datasets & DataLoaders
# ─────────────────────────────────────────────────────────────────────
print("\n[2/4] Building datasets...")
train_ds = LineDataset(
    csv_file   = CFG["data"]["train_csv"],
    img_dir    = CFG["data"]["train_img_dir"],
    tokenizer  = tokenizer,
    transform  = train_transform,
    max_length = CFG["model"]["max_length"],
)
val_ds = LineDataset(
    csv_file   = CFG["data"]["val_csv"],
    img_dir    = CFG["data"]["val_img_dir"],
    tokenizer  = tokenizer,
    transform  = val_transform,
    max_length = CFG["model"]["max_length"],
)

NW = CFG["data"]["num_workers"]
train_loader = DataLoader(
    train_ds,
    batch_size  = CFG["training"]["batch_size"],
    shuffle     = True,
    num_workers = NW,
    collate_fn  = collate_fn,
    pin_memory  = True,
    drop_last   = True,
)
val_loader = DataLoader(
    val_ds,
    batch_size  = CFG["training"]["batch_size"] * 2,
    shuffle     = False,
    num_workers = NW,
    collate_fn  = collate_fn,
    pin_memory  = True,
)
print(f"Train : {len(train_ds):,}   Val : {len(val_ds):,}")

# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
print("\n[3/4] Building model...")
model = build_model(
    encoder_name   = CFG["model"]["encoder_name"],
    decoder_name   = CFG["model"]["decoder_name"],
    tokenizer      = tokenizer,
    max_length     = CFG["model"]["max_length"],
    dropout        = CFG["model"]["dropout"],
    freeze_encoder = True,
)
model.to(device)

counts = count_parameters(model)
print(f"Encoder  : {counts['encoder']:,} params")
print(f"Decoder  : {counts['decoder']:,} params")
print(f"Total    : {counts['total']:,} params")
print(f"Trainable: {counts['trainable']:,} params (encoder frozen)")

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"\nResuming from {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────
print("\n[4/4] Training...")
trainer = Trainer(
    model            = model,
    train_loader     = train_loader,
    val_loader       = val_loader,
    tokenizer        = tokenizer,
    config           = CFG,
    device           = device,
    output_dir       = OUTPUT_DIR,
    accumulate_steps = ACCUM_STEPS,
)
best_cer = trainer.run()

print(f"\n✓ Training complete! Best CER : {best_cer:.4f}")
print(f"  Checkpoints in : {OUTPUT_DIR}")
