"""
evaluate.py — Offline evaluation script for Bangla OCR V2.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
                       --val_csv /path/to/val.csv
                       --img_dir /path/to/images
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, ViTImageProcessor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import ImageTransform
from src.data.dataset import LineDataset, collate_fn
from src.models.ocr_model import build_model
from src.training.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Bangla OCR V2 — Evaluation")
    parser.add_argument("--checkpoint",   required=True,  help="Path to best_model.pt")
    parser.add_argument("--val_csv",      required=True,  help="Path to val.csv")
    parser.add_argument("--img_dir",      required=True,  help="Directory containing images")
    parser.add_argument("--encoder",      default="google/vit-base-patch16-224")
    parser.add_argument("--decoder",      default="xlm-roberta-base")
    parser.add_argument("--max_length",   type=int, default=128)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_beams",    type=int, default=4)
    parser.add_argument("--print_n",      type=int, default=10,
                        help="Number of sample predictions to print")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    processor = ViTImageProcessor.from_pretrained(args.encoder)
    transform = ImageTransform(processor, augment=False)

    dataset = LineDataset(
        csv_file   = args.val_csv,
        img_dir    = args.img_dir,
        tokenizer  = tokenizer,
        transform  = transform,
        max_length = args.max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(args.encoder, args.decoder, tokenizer,
                        args.max_length, freeze_encoder=False)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}\n")

    metrics = evaluate_model(
        model, loader, tokenizer, device,
        num_beams=args.num_beams,
        num_samples_to_print=args.print_n,
    )

    print("\n" + "=" * 50)
    print(f"CER  : {metrics['cer']:.4f}   ({(1-metrics['cer'])*100:.2f}% char accuracy)")
    print(f"WER  : {metrics['wer']:.4f}   ({(1-metrics['wer'])*100:.2f}% word accuracy)")
    print(f"Acc  : {metrics['exact_match_accuracy']*100:.2f}% exact match")
    print("=" * 50)


if __name__ == "__main__":
    main()
