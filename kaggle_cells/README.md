# Kaggle Cells Guide

This directory contains standalone python scripts that are meant to be run directly in Kaggle Notebook cells. They should be copied into code cells and run in sequential order.

## `01_setup.py`
**Purpose:** Sets up the environment.
Installs necessary pip packages (`jiwer`, `openpyxl`, `albumentations`).
Checks GPU availability and memory.
If the repository is attached as a Kaggle input dataset, it copies it to the working directory (`/kaggle/working/bangla-ocr-roberta`) to allow modifying scripts and saving outputs.
Adds the project root to `sys.path`.

## `02_prepare_data.py`
**Purpose:** Formats the BN-HTRd dataset for training.
Reads `.xlsx` files to get ground truth text.
Groups word-level text into line-level text.
Matches lines with corresponding `.jpg` image files.
Copies the valid images to a flat directory (`/kaggle/working/data/train/`) to avoid read-only limitations.
Splits data into train (90%) and validation (10%) sets, saving them as `train.csv` and `val.csv`.

## `03_train.py`
**Purpose:** Runs the main training loop.
Defines the hyperparameter configuration (`CFG`).
Initializes the tokenizer and ViT image processor.
Creates data loaders for the train and validation sets.
Builds the `VisionEncoderDecoderModel` (ViT-384 + XLM-RoBERTa).
Initializes the `Trainer` class and starts training.
Saves model checkpoints (`last_model.pt` and `best_model.pt`) to the working directory.

## `04_evaluate.py`
**Purpose:** Evaluates the best model with high precision.
Loads the `best_model.pt` checkpoint.
Performs beam search decoding (`num_beams=4`) on the validation set for accurate evaluation (this is slower than the greedy decoding used during training).
Calculates and prints the final Character Error Rate (CER), Word Error Rate (WER), and exact match accuracy.
Prints a few sample predictions to visually inspect quality.
