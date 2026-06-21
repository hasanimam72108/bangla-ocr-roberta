"""
ocr_model.py — ViT + XLM-RoBERTa model builder for Bangla OCR V2.

Architecture overview:
  Encoder : google/vit-base-patch16-224  (86 M params)
  Decoder : xlm-roberta-base             (278 M params, cross-attention added)
  Glue    : HuggingFace VisionEncoderDecoderModel

Why XLM-RoBERTa over TrOCR's decoder?
  TrOCR uses a GPT-2 style decoder whose vocabulary and byte-pair merges are
  English-biased.  XLM-RoBERTa was pre-trained on 2.5 TB of multilingual text
  including Bengali, so its SentencePiece tokenizer natively handles every
  Unicode character and conjunct used in Bangla script.
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTModel,
    ViTConfig,
    AutoConfig,
    AutoModelForCausalLM,
    XLMRobertaConfig,
)


# ──────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────

def build_model(
    encoder_name: str = "google/vit-base-patch16-224",
    decoder_name: str = "xlm-roberta-base",
    tokenizer=None,
    max_length: int = 128,
    dropout: float = 0.1,
    freeze_encoder: bool = True,
) -> VisionEncoderDecoderModel:
    """
    Build and configure a VisionEncoderDecoderModel.

    Steps:
      1. Load ViT encoder from pre-trained weights.
      2. Load XLM-RoBERTa config, patch it for cross-attention decoding.
      3. Instantiate VisionEncoderDecoderModel from encoder+decoder pretrained.
      4. Wire up all special token ids across model / generation config.
      5. Optionally freeze encoder parameters.

    Args:
        encoder_name  : HuggingFace model hub id for the ViT encoder.
        decoder_name  : HuggingFace model hub id for the XLM-R decoder.
        tokenizer     : XLMRobertaTokenizer (or AutoTokenizer) instance.
        max_length    : Maximum generation length (tokens).
        dropout       : Dropout probability applied to the decoder.
        freeze_encoder: If True, freeze all encoder parameters so only the
                        decoder + cross-attention layers train in phase-1.

    Returns:
        Configured VisionEncoderDecoderModel ready for .train() / .generate().
    """
    print(f"  Loading encoder  : {encoder_name}")
    print(f"  Loading decoder  : {decoder_name}")

    # from_encoder_decoder_pretrained handles the cross-attention injection
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name,
        decoder_name,
        encoder_add_pooling_layer=False,
    )

    # ── Special token wiring ──────────────────
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided to configure special tokens.")

    bos_id = tokenizer.cls_token_id   # XLM-R uses <s> / cls as BOS
    eos_id = tokenizer.sep_token_id   # XLM-R uses </s> / sep as EOS
    pad_id = tokenizer.pad_token_id

    # Top-level model config
    model.config.decoder_start_token_id = bos_id
    model.config.bos_token_id           = bos_id
    model.config.eos_token_id           = eos_id
    model.config.pad_token_id           = pad_id
    model.config.max_length             = max_length

    # Decoder sub-config
    model.config.decoder.is_decoder             = True
    model.config.decoder.add_cross_attention    = True
    model.config.decoder.decoder_start_token_id = bos_id
    model.config.decoder.bos_token_id           = bos_id
    model.config.decoder.eos_token_id           = eos_id
    model.config.decoder.pad_token_id           = pad_id

    # Generation config (used by .generate())
    model.generation_config.decoder_start_token_id = bos_id
    model.generation_config.bos_token_id           = bos_id
    model.generation_config.eos_token_id           = eos_id
    model.generation_config.pad_token_id           = pad_id
    model.generation_config.max_length             = max_length

    # ── Optional dropout override ─────────────
    if dropout != 0.1:
        for module in model.decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    # ── Freeze encoder ────────────────────────
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen — only decoder + cross-attn weights will train.")

    return model


# ──────────────────────────────────────────────
# Phased un-freezing helper
# ──────────────────────────────────────────────

def unfreeze_encoder(model: VisionEncoderDecoderModel,
                     optimizer: torch.optim.Optimizer,
                     encoder_lr: float = 5e-6) -> None:
    """
    Un-freeze all encoder parameters and add them to the optimizer
    with a separate (smaller) learning rate group.

    Call this after the initial frozen-encoder warm-up phase.
    """
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Add a new param group for the encoder with a much smaller LR
    optimizer.add_param_group({
        "params": list(model.encoder.parameters()),
        "lr": encoder_lr,
        "name": "encoder",
    })
    print(f"  Encoder un-frozen. Added to optimizer with lr={encoder_lr:.2e}")


# ──────────────────────────────────────────────
# Parameter accounting
# ──────────────────────────────────────────────

def count_parameters(model: VisionEncoderDecoderModel) -> dict:
    encoder_params  = sum(p.numel() for p in model.encoder.parameters())
    decoder_params  = sum(p.numel() for p in model.decoder.parameters())
    total_params    = encoder_params + decoder_params
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "encoder":   encoder_params,
        "decoder":   decoder_params,
        "total":     total_params,
        "trainable": trainable,
    }
