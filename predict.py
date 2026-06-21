"""
predict.py — Single-image inference for Bangla OCR V2.

Usage:
    python predict.py --image path/to/line.jpg
                      --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessing import ImageTransform, preprocess_cv, load_image
from src.models.ocr_model import build_model


def predict_single(
    image_path: str,
    model,
    tokenizer,
    transform: ImageTransform,
    device: torch.device,
    num_beams: int = 4,
    max_new_tokens: int = 128,
) -> str:
    """Run inference on a single line image and return the predicted text."""
    try:
        cv_img  = load_image(image_path)
        pil_img = preprocess_cv(cv_img)
    except Exception:
        pil_img = Image.open(image_path).convert("RGB")

    pixel_values = transform(pil_img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Bangla OCR V2 — Single Image Prediction")
    parser.add_argument("--image",       required=True,  help="Path to the line image")
    parser.add_argument("--checkpoint",  required=True,  help="Path to best_model.pt")
    parser.add_argument("--encoder",     default="google/vit-base-patch16-224")
    parser.add_argument("--decoder",     default="xlm-roberta-base")
    parser.add_argument("--max_length",  type=int, default=128)
    parser.add_argument("--num_beams",   type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    processor = ViTImageProcessor.from_pretrained(args.encoder)
    transform = ImageTransform(processor, augment=False)

    model = build_model(args.encoder, args.decoder, tokenizer,
                        args.max_length, freeze_encoder=False)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    prediction = predict_single(
        args.image, model, tokenizer, transform, device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_length,
    )
    print(f"Predicted text: {prediction}")


if __name__ == "__main__":
    main()
