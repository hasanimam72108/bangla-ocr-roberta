"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE CELL 01 — Environment Setup                                  ║
║  Paste the body of this script (inside __name__=='__main__') into    ║
║  the first code cell of your Kaggle notebook.                        ║
╚══════════════════════════════════════════════════════════════════════╝

Run once at the start of every Kaggle session.
"""

# ── Install extra packages ───────────────────────────────────────────
import subprocess, sys

packages = [
    "jiwer>=3.0.0",
    "openpyxl",          # needed by pandas to read .xlsx
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
# If you added this repo as a Kaggle dataset, copy it:
import shutil, os

REPO_INPUT = "/kaggle/input/bangla-ocr-roberta"   # adjust to your dataset slug
WORK_DIR   = "/kaggle/working/bangla-ocr-roberta"

if os.path.exists(REPO_INPUT) and not os.path.exists(WORK_DIR):
    shutil.copytree(REPO_INPUT, WORK_DIR)
    print(f"Copied repo to {WORK_DIR}")
elif not os.path.exists(REPO_INPUT):
    print("⚠️  Repo dataset not attached — make sure to add it via 'Add Input'.")

# Add project root to sys.path so imports work
import sys
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print("\n✓ Setup complete.")
