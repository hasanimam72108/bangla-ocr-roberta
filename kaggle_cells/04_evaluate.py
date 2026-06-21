"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE CELL 04 — Evaluation / Inference on saved checkpoint         ║
╚══════════════════════════════════════════════════════════════════════╝

Load best_model.pt and run full CER / WER metrics on the validation set.
Also shows N decoded sample predictions so you can eyeball quality.
"""

import os, sys
PROJ = "/kaggle/working/bangla-ocr-roberta"
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import torch
from transformers import AutoTokenizer, ViTImageProcessor
from torch.utils.data import DataLoader

from src.data.preprocessing import ImageTransform
from src.data.dataset        import LineDataset, collate_fn
from src.models.ocr_model    import build_model
from src.training.metrics    import evaluate_model

# ─────────────────────────────────────────────────────────────────────
# ✏️  Settings
# ─────────────────────────────────────────────────────────────────────
ENCODER_NAME = "google/vit-base-patch16-224"
DECODER_NAME = "xlm-roberta-base"
CHECKPOINT   = "/kaggle/working/checkpoints/best_model.pt"
VAL_CSV      = "/kaggle/working/data/val.csv"
IMG_DIR      = "/kaggle/working/data/train"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
NUM_BEAMS    = 4
PRINT_SAMPLES= 10

# ─────────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(DECODER_NAME)
processor = ViTImageProcessor.from_pretrained(ENCODER_NAME)
transform = ImageTransform(processor, augment=False)

val_ds = LineDataset(
    csv_file   = VAL_CSV,
    img_dir    = IMG_DIR,
    tokenizer  = tokenizer,
    transform  = transform,
    max_length = MAX_LENGTH,
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

model = build_model(ENCODER_NAME, DECODER_NAME, tokenizer, MAX_LENGTH,
                    freeze_encoder=False)
model.to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

metrics = evaluate_model(
    model, val_loader, tokenizer, device,
    num_beams=NUM_BEAMS,
    num_samples_to_print=PRINT_SAMPLES,
)

print("\n" + "=" * 50)
print(f"Character Error Rate  (CER) : {metrics['cer']:.4f}  ({(1-metrics['cer'])*100:.2f}% char accuracy)")
print(f"Word Error Rate       (WER) : {metrics['wer']:.4f}  ({(1-metrics['wer'])*100:.2f}% word accuracy)")
print(f"Exact Match Accuracy        : {metrics['exact_match_accuracy']*100:.2f}%")
print("=" * 50)
