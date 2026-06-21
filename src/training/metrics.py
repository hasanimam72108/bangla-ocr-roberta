"""
metrics.py — CER / WER / Exact-Match evaluation for Bangla OCR V2.

Changes from V1:
  - generate() now uses beam-search (num_beams is configurable) for a more
    realistic quality estimate during validation.
  - We skip the autocast on CPU paths to avoid the deprecation warning.
  - The function also collects and returns a few decoded sample strings so the
    trainer can log them for quick visual inspection.
"""

import torch
from tqdm import tqdm
from jiwer import cer, wer


# ──────────────────────────────────────────────
# Scalar metrics
# ──────────────────────────────────────────────

def compute_cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate (0 = perfect)."""
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    return cer(ground_truth, prediction)


def compute_wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate (0 = perfect)."""
    if not ground_truth:
        return 0.0 if not prediction else 1.0
    return wer(ground_truth, prediction)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction.strip() == ground_truth.strip()


# ──────────────────────────────────────────────
# Full validation loop
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    num_samples_to_print: int = 3,
) -> dict:
    """
    Run greedy / beam-search decoding over the validation set and compute
    aggregate CER, WER and exact-match accuracy.

    Args:
        model            : VisionEncoderDecoderModel (in eval mode).
        dataloader       : Validation DataLoader.
        tokenizer        : XLM-R tokenizer.
        device           : Torch device.
        num_beams        : Beam width for generation (1 = greedy).
        max_new_tokens   : Hard cap on generated token count.
        num_samples_to_print: How many GT/Pred pairs to print for visual check.

    Returns:
        dict with keys: cer, wer, exact_match_accuracy, samples
    """
    model.eval()

    # Ensure generation config is consistent
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id   # </s>
    model.config.pad_token_id            = tokenizer.pad_token_id
    model.config.eos_token_id            = tokenizer.sep_token_id

    is_cuda       = "cuda" in str(device)
    total_cer     = 0.0
    total_wer     = 0.0
    exact_matches = 0
    total         = 0
    sample_pairs  = []   # [(gt, pred), ...]

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        texts        = batch["texts"]

        # Use autocast only on CUDA
        ctx = torch.autocast("cuda", dtype=torch.float16) if is_cuda else torch.no_grad()
        with ctx:
            generated_ids = model.generate(
                pixel_values,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
            )

        for i, gen_ids in enumerate(generated_ids):
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gt   = texts[i].strip()

            total_cer     += compute_cer(pred, gt)
            total_wer     += compute_wer(pred, gt)
            if compute_exact_match(pred, gt):
                exact_matches += 1
            total += 1

            if len(sample_pairs) < num_samples_to_print:
                sample_pairs.append((gt, pred))

    # Print samples for visual inspection
    print("\n--- Validation Samples ---")
    for k, (gt, pred) in enumerate(sample_pairs, 1):
        print(f"  [{k}] GT  : {gt}")
        print(f"      PRED: {pred}")
    print("-" * 26)

    avg_cer  = total_cer  / max(total, 1)
    avg_wer  = total_wer  / max(total, 1)
    accuracy = exact_matches / max(total, 1)

    return {
        "cer":                 avg_cer,
        "wer":                 avg_wer,
        "exact_match_accuracy": accuracy,
        "samples":             sample_pairs,
    }
