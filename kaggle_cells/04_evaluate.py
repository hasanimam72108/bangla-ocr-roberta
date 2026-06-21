"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE CELL 04 — Final Quality Evaluation                           ║
╚══════════════════════════════════════════════════════════════════════╝

Run this cell AFTER Cell 03 finishes (or after resuming a session).

This is the ONLY place that uses beam=4 decoding, which gives the real
accuracy numbers but takes ~15-20 min.  Cell 03 used greedy (beam=1)
during training to save time; the trained weights are identical either way.

Outputs: CER, WER, exact-match accuracy + N sample GT vs Predicted pairs.
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
ENCODER_NAME = "google/vit-base-patch16-384"  # must match what Cell 03 trained
DECODER_NAME = "xlm-roberta-base"
CHECKPOINT   = "/kaggle/working/checkpoints/best_model.pt"
VAL_CSV      = "/kaggle/working/data/val.csv"
IMG_DIR      = "/kaggle/working/data/train"
MAX_LENGTH   = 96     # must match Cell 03
BATCH_SIZE   = 16     # lower batch for beam search (more VRAM per sample)
NUM_BEAMS    = 4      # ⭐ beam=4 here for real quality numbers (Cell 03 used beam=1)
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
