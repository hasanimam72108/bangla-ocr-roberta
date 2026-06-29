# Bangla Handwritten OCR V2 — ViT + XLM-RoBERTa

A robust Bengali handwriting recognition pipeline combining:
- **Encoder**: `google/vit-base-patch16-224` — Vision Transformer
- **Decoder**: `xlm-roberta-base` — Multilingual language model, natively pre-trained on Bengali text
- **Bridge**: HuggingFace `VisionEncoderDecoderModel`

## Why the upgrade from V1 (TrOCR)?

TrOCR's decoder is GPT-2 based with English-biased BPE merges, causing frequent out-of-vocabulary failures on Bengali conjuncts (যুক্তাক্ষর). XLM-RoBERTa uses SentencePiece which handles every Bengali Unicode codepoint natively.

---

## Project Structure

```
bangla-ocr-roberta/
├── configs/
│   └── train_config.yaml          # All hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py             # LineDataset + collate_fn
│   │   └── preprocessing.py       # CLAHE + ViTImageProcessor + augmentation
│   ├── models/
│   │   └── ocr_model.py           # build_model(), unfreeze_encoder(), count_parameters()
│   └── training/
│       ├── trainer.py             # Trainer with phased training + grad accumulation
│       └── metrics.py             # CER, WER, exact-match
├── kaggle_cells/
│   ├── 01_setup.py                # Cell 1: env setup
│   ├── 02_prepare_data.py         # Cell 2: data alignment from BN-HTRd xlsx
│   ├── 03_train.py                # Cell 3: training
│   └── 04_evaluate.py             # Cell 4: evaluation
├── train.py                       # CLI training entry-point
├── evaluate.py                    # CLI evaluation entry-point
├── predict.py                     # Single-image inference
└── requirements.txt
```

---

## Quick Start (Kaggle)

1. **Add Inputs** to your Kaggle notebook:
   - This repository (as a Code dataset)
   - The BN-HTRd dataset

2. Run cells in order:
   - Paste `kaggle_cells/01_setup.py` → Run
   - Paste `kaggle_cells/02_prepare_data.py` → Run (adjust `BASE_DIR`)
   - Paste `kaggle_cells/03_train.py` → Run

3. Checkpoints are saved to `/kaggle/working/checkpoints/`:
   - `last_model.pt` — overwritten every epoch (disk safe)
   - `best_model.pt` — only updated on CER improvement

---

## Training Strategy

| Phase | Epochs | What trains |
|-------|--------|-------------|
| Warm-up | 1–5 | Decoder + cross-attention only (encoder frozen) |
| Fine-tune | 6+ | Full model, encoder at 5e-6 LR |

---

## Local Usage

```bash
pip install -r requirements.txt

# Prepare data (set paths inside the script)
python scripts/prepare_data.py

# Train
python train.py --config configs/train_config.yaml --output_dir checkpoints/

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt \
                   --val_csv data/val.csv \
                   --img_dir data/train

# Single image
python predict.py --image sample.jpg --checkpoint checkpoints/best_model.pt
```

---

## Authors

- **Hasan Imam** ([GitHub: hasanimam72108](https://github.com/hasanimam72108))
- **Jawadur Rafid** ([GitHub: jawadur13](https://github.com/jawadur13))