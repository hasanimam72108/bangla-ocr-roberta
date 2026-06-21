# Accuracy Improvement Suggestions — Bangla OCR V2

> Suggestions **beyond what is already in the architecture plan** that could meaningfully improve CER/WER on the BN-HTRd benchmark.

---

## 1. 🔄 Use a Larger ViT Encoder (High Impact, Easy Swap)

**What**: Replace `vit-base-patch16-224` with `google/vit-large-patch16-224` or `google/vit-base-patch16-384`.

**Why**: The patch-16-384 variant uses a higher input resolution (384×384) which preserves more detail for dense scripts like Bangla. Handwritten conjuncts (যুক্তাক্ষর) span multiple pixels and get compressed unfavorably at 224px.

**Change needed**:
```yaml
# train_config.yaml
model:
  encoder_name: "google/vit-base-patch16-384"
  image_size: 384
```
> ⚠️ Requires ~2× more GPU memory than 224. Reduce batch_size or increase accumulate_steps.

---

## 2. 🧠 Use a Stronger Decoder (High Impact, Low Risk)

**What**: Swap `xlm-roberta-base` for `facebook/mbart-large-cc25` (Bengali is a supported language, code: `bn_IN`).

**Why**: mBART was pre-trained with an **encoder-decoder** objective on 25 languages including Bengali, so it already "knows" Bengali at both the token and sentence level. XLM-RoBERTa is encoder-only and is adapted as a decoder — it works, but isn't its native role.

**Note**: mBART's tokenizer uses a `<lang_code>` forced BOS token, which requires a small change in the tokenizer setup and `decoder_start_token_id`.

---

## 3. 🔡 Character-Level Beam Search with Language Model Rescoring (Medium Impact)

**What**: After beam-search decoding, rerank candidate strings using a Bengali n-gram or character-level language model.

**Why**: The decoder can produce tokens that are visually plausible but linguistically impossible (e.g., illegal vowel sequences). A shallow language model catches these.

**Implementation**: Train a KenLM model on a Bengali text corpus (e.g. Oscar/CC-100 Bengali) and integrate it with `pyctcdecode` or a simple log-probability rescorer.

---

## 4. 📐 Aspect-Ratio-Preserving Resize (Medium Impact)

**What**: Instead of squashing the line image to a 224×224 square, resize only the height to 224 and pad the width.

**Why**: Bengali handwritten lines are typically wide (aspect ≈ 3:1 to 8:1). Squashing to a square introduces horizontal compression that distorts character shapes, making it harder for the ViT to distinguish visually similar glyphs (e.g., ব vs র).

**Trade-off**: ViT requires a fixed square input. The cleanest solution is to use **DeiT** or **BEiT** with a flexible sequence length, or tile the image into overlapping 224×224 patches and aggregate patch features.

---

## 5. 🪄 CTC Head as an Auxiliary Loss (Medium Impact, More Complex)

**What**: Add a CTC (Connectionist Temporal Classification) head on top of the ViT encoder output in parallel with the autoregressive decoder. Use a weighted sum of CTC loss + cross-entropy loss during training.

**Why**: CTC provides a strong monotonic alignment signal that complements the autoregressive decoder, especially in early training epochs when the cross-attention layers have not yet converged. This is the approach used by **Conformer-based ASR systems** and has been successfully applied to OCR.

**Implementation sketch**:
```python
ctc_logits = self.ctc_head(encoder_output)   # (B, T, vocab)
ctc_loss = F.ctc_loss(ctc_logits.log_softmax(-1).permute(1,0,2), ...)
total_loss = 0.7 * ce_loss + 0.3 * ctc_loss
```

---

## 6. 📊 Label Smoothing Increase for Ambiguous Conjuncts

**What**: Increase `label_smoothing` from 0.1 to 0.15–0.2.

**Why**: Bengali handwriting datasets have high inter-writer variability. A slightly higher label-smoothing value teaches the model to be less overconfident on ambiguous glyph forms that overlap visually (e.g., ত vs ৎ, ড vs ড়).

---

## 7. 🎭 Script-Aware Augmentation (Bangla-Specific, Medium Impact)

**What**: Add ink-simulation augmentation specific to pen/pencil strokes:
- **Elastic deformation** — simulates natural handwriting variability
- **Dilation/erosion morphology** — simulates thick/thin pen styles
- **Grid distortion** — simulates paper warp

**Library**: `albumentations` has `ElasticTransform` and `GridDistortion` which are significantly better than `RandomPerspective` for script-level distortions.

```python
import albumentations as A
aug = A.Compose([
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.4),
    A.GridDistortion(p=0.3),
    A.Morphological(scale=(1, 2), operation="dilation", p=0.2),
])
```

---

## 8. 🔄 Curriculum Learning (Medium Impact, Training-Time Only)

**What**: Sort training samples by text length (ascending) for the first 10 epochs, then shuffle randomly.

**Why**: Short lines (1–3 words) are easier and help the model learn alignment quickly. Introducing long lines too early increases loss variance and can destabilize cross-attention training.

**Implementation**: Sort `train.csv` by `text.str.len()`, partition into easy/medium/hard buckets, and schedule the DataLoader accordingly.

---

## 9. 🌐 Pre-training on Synthetic Bangla Data

**What**: Generate synthetic line images using a variety of Bengali Unicode fonts + random backgrounds and pre-train the model on these before fine-tuning on BN-HTRd.

**Why**: BN-HTRd has ~50K lines. Synthetic data can be generated in millions, giving the model a strong prior on Bengali character shapes before seeing real handwriting.

**Tools**: Python `Pillow` + `freetype-py` + Bengali font packs from Google Fonts.

---

## Priority Matrix

| Suggestion | Expected Gain | Implementation Effort | GPU Cost |
|---|---|---|---|
| ViT-384 encoder | **High** | Low | High |
| mBART decoder | **High** | Medium | Medium |
| Aspect-ratio resize | Medium | Medium | Low |
| Script-aware augmentation | Medium | Low | Low |
| CTC auxiliary loss | Medium | High | Low |
| LM rescoring | Medium | High | Very Low |
| Curriculum learning | Low-Medium | Low | None |
| Synthetic pre-training | **High** | Very High | Very High |
| Label smoothing increase | Low | Trivial | None |
