# KAGGLE CELL 01 — Environment Setup

# ── Install extra packages ───────────────────────────────────────────
import subprocess, sys

packages = [
    "jiwer>=3.0.0",
    "openpyxl",              # needed by pandas to read .xlsx
    "albumentations>=1.3.1", # Suggestion 7: elastic/grid distortion augmentations
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── Verify GPU ───────────────────────────────────────────────────────
import torch
print("PyTorch  :", torch.__version__)
print("CUDA     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU      :", torch.cuda.get_device_name(0))
    print("VRAM     :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# ── Clone / copy project files ───────────────────────────────────────
import os
import sys
import shutil
import subprocess

WORK_DIR = "/kaggle/working/bangla-ocr-roberta"
REPO_INPUT = "/kaggle/input/bangla-ocr-roberta"   # if attached as a Kaggle dataset

if not os.path.exists(WORK_DIR):
    if os.path.exists(REPO_INPUT):
        print(f"\nCopying repository from Kaggle input {REPO_INPUT}...")
        shutil.copytree(REPO_INPUT, WORK_DIR)
        print(f"✓ Copied to {WORK_DIR}")
    else:
        GITHUB_REPO_URL = "https://github.com/hasanimam72108/bangla-ocr-roberta.git"
        print(f"\nCloning repository from {GITHUB_REPO_URL}...")
        subprocess.check_call(["git", "clone", GITHUB_REPO_URL, WORK_DIR])
        print(f"✓ Cloned to {WORK_DIR}")
else:
    print(f"\n✓ Repository already exists at {WORK_DIR}")
    # Optional: pull latest changes if you restart the session
    # subprocess.check_call(["git", "-C", WORK_DIR, "pull"])

# Add project root to sys.path so imports work
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print("\n✓ Setup complete.")
