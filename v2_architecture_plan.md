# Bangla OCR V2: ViT + XLM-RoBERTa Hybrid Model Plan

## Context for the AI
Hello! The user is migrating from a previous TrOCR-based Bengali handwriting OCR pipeline to a more robust **ViT + XLM-RoBERTa hybrid model**. The previous project is located at `D:\Projects\-bangla-handwritten-ocr`. 
Your job is to recreate the pipeline in this new repository, maintaining the modular structure and generating paste-ready Kaggle cell scripts, while implementing the new architecture.

## 1. The Model Architecture
We are abandoning the `microsoft/trocr-base-handwritten` model because its decoder is English-first and struggles with out-of-vocabulary Bengali words.
- **Encoder**: `google/vit-base-patch16-224` (Vision Transformer).
- **Decoder**: `xlm-roberta-base` (Multilingual language model, pre-trained on Bengali).
- **Integration**: Use HuggingFace's `VisionEncoderDecoderModel.from_encoder_decoder_pretrained()`.
- **Tokenizer**: Use `XLMRobertaTokenizer` (or AutoTokenizer from `xlm-roberta-base`). You will *not* need the custom `BanglaGraphemeTokenizer` used in the old project, as XLM-RoBERTa already handles Bengali efficiently.

## 2. The Dataset & Kaggle Environment
**IMPORTANT CHANGE:** In the previous project, we downloaded the dataset via an HTTP link. In this project, the user will attach the BN-HTRd dataset directly to the Kaggle notebook using the **"Add Input"** method. 

Assume the dataset is located at: `/kaggle/input/bn-htrd/` (or similar). Do not write code to download or unzip the dataset.

### Dataset Alignment Logic (CRITICAL)
The dataset contains a physical misalignment issue if you just iterate through folders. You **MUST** align the data using the provided `.xlsx` ground truth files.
1. **Images Location:** Line crops are located somewhere like `/kaggle/input/bn-htrd/words/` or within subfolders.
2. **Ground Truths:** Excel files are provided per document (e.g., `129.xlsx`) in a `ground_truths` folder.
3. **The Mapping:** Each row in the Excel file has an `Id` column (formatted as `doc_writer_line_word`, e.g., `129_1_19_1`) and a `word` column (the Bengali text).
4. **Data Preparation Script:** You must write a `prepare_data.py` script that iterates over the `.xlsx` files, groups the words by their line (e.g., grouping `129_1_19_1` and `129_1_19_2` into line `129_1_19`), joins the Bengali text with spaces, and matches it to the corresponding image file `129_1_19.jpg`. 
*(Hint: You can look at `D:\Projects\-bangla-handwritten-ocr\scripts\prepare_data.py` to see exactly how this was solved previously).*

## 3. Required Outputs
Please generate the following structure for the user:
1. **Core Pipeline Files:** `dataset.py`, `preprocessing.py`, `train.py`, `metrics.py`, and `train_config.yaml`.
2. **Kaggle Cell Scripts:** Provide numbered scripts (e.g., `01_setup.py`, `02_prepare_data.py`, `03_train.py`) that the user can directly copy and paste into Kaggle cells.
3. **Training Fixes:** 
   - Ensure the Kaggle output disk doesn't fill up! Save checkpoints by constantly overwriting `last_model.pt` and `best_model.pt` rather than saving a new file every epoch.
   - The dataloader will expect images in a single folder. If the script copies images, copy them to `/kaggle/working/data/train/` to avoid read-only errors from the `/kaggle/input/` directory.

## 4. Immediate Next Steps for the AI
1. Review the old repository path provided by the user to understand the codebase flow.
2. Scaffold the new project directory.
3. Write the `VisionEncoderDecoderModel` integration.
4. Provide the user with the paste-ready Kaggle code cells to begin training!
