# KAGGLE CELL 02 — Data Preparation

import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────
# ✏️  Auto-discover the dataset path in Kaggle
# ─────────────────────────────────────────────────────────────────────
def find_dataset_base_dir():
    search_path = "/kaggle/input"
    if not os.path.exists(search_path):
        # Fallback for local testing if needed
        return "./bn-htrd"
        
    # We use os.walk but prune massive image directories in-place.
    # This makes the search blazing fast while working at any depth.
    for root, dirs, files in os.walk(search_path):
        if "Recognition_Ground_Truth_Texts" in dirs and "Segmentation_Images" in dirs:
            return root
            
        # Modifying 'dirs' in-place tells os.walk NOT to traverse into these
        for skip_dir in ["Segmentation_Images", "Lines", "train", "test", ".git", "Sample_Small"]:
            if skip_dir in dirs:
                dirs.remove(skip_dir)

    return None

BASE_DIR = find_dataset_base_dir()

if BASE_DIR is None:
    # Print available directories to help the user debug
    inputs = os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else []
    raise FileNotFoundError(
        f"Could not find 'Recognition_Ground_Truth_Texts' in /kaggle/input/.\n"
        f"Available folders in /kaggle/input/ are: {inputs}\n"
        f"Please make sure the BN-HTRd dataset is attached to the notebook!"
    )

print(f"Dataset found at: {BASE_DIR}")

TEXT_DIR  = os.path.join(BASE_DIR, "Recognition_Ground_Truth_Texts")
LINES_DIR = os.path.join(BASE_DIR, "Segmentation_Images", "Lines")


OUT_DIR   = "/kaggle/working/data"
IMG_DIR   = os.path.join(OUT_DIR, "train")   # all images go here (flat)

os.makedirs(IMG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Helper: try common extensions
# ─────────────────────────────────────────────────────────────────────
_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def find_image(directory: str, stem: str) -> str | None:
    for ext in _EXTENSIONS:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────
# Main data preparation loop
# ─────────────────────────────────────────────────────────────────────
def prepare_data() -> pd.DataFrame:
    doc_folders = sorted([
        d for d in os.listdir(TEXT_DIR)
        if os.path.isdir(os.path.join(TEXT_DIR, d))
    ])
    print(f"Found {len(doc_folders)} document folders.")

    records = []

    for doc_id in tqdm(doc_folders, desc="Processing docs"):
        xlsx_path = os.path.join(TEXT_DIR, doc_id, f"{doc_id}.xlsx")
        if not os.path.exists(xlsx_path):
            continue

        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
        except Exception as e:
            print(f"  ⚠️  Could not read {xlsx_path}: {e}")
            continue

        # Normalize column names (some versions use 'word' vs 'Word')
        df.columns = [c.strip() for c in df.columns]
        col_map    = {c.lower(): c for c in df.columns}
        if "id" not in col_map or "word" not in col_map:
            print(f"  ⚠️  Missing Id/Word columns in {xlsx_path}, skipping.")
            continue

        df = df.rename(columns={col_map["id"]: "Id", col_map["word"]: "Word"})
        df = df.dropna(subset=["Id", "Word"])
        df["Id"]   = df["Id"].astype(str).str.strip()
        df["Word"] = df["Word"].astype(str).str.strip()

        # ── Group words into lines ────────────────────────────────────
        # Id format: doc_writer_line_word  (e.g. "129_1_19_3")
        # line_id  : doc_writer_line       (e.g. "129_1_19")
        df["line_id"] = df["Id"].apply(lambda x: x.rsplit("_", 1)[0])

        # sort=False preserves the original word order in the sheet
        line_df = (
            df.groupby("line_id", sort=False)["Word"]
              .apply(lambda words: " ".join(words))
              .reset_index()
        )

        # ── Match each line to its image file ─────────────────────────
        for _, row in line_df.iterrows():
            line_id = row["line_id"]
            text    = row["Word"].strip()
            if not text:
                continue

            parts = line_id.split("_")
            if len(parts) < 3:
                continue

            current_doc = parts[0]
            writer_id   = f"{parts[0]}_{parts[1]}"
            img_dir     = os.path.join(LINES_DIR, current_doc, writer_id)

            src_path = find_image(img_dir, line_id)
            if src_path is None:
                continue

            # Copy image to flat output dir (avoids read-only /kaggle/input issues)
            dest_name = f"{line_id}.jpg"
            dest_path = os.path.join(IMG_DIR, dest_name)
            if not os.path.exists(dest_path):          # skip if already copied
                shutil.copy2(src_path, dest_path)

            records.append({"image": dest_name, "text": text})

    df_all = pd.DataFrame(records)
    print(f"\nTotal aligned pairs  : {len(df_all)}")
    return df_all


# ─────────────────────────────────────────────────────────────────────
# Split & save
# ─────────────────────────────────────────────────────────────────────
df_all = prepare_data()

if len(df_all) == 0:
    raise RuntimeError(
        "No data was prepared. Check that BASE_DIR points to the correct "
        "dataset mount path and that the xlsx files exist."
    )

train_df, val_df = train_test_split(df_all, test_size=0.10, random_state=42)

train_csv = os.path.join(OUT_DIR, "train.csv")
val_csv   = os.path.join(OUT_DIR, "val.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv,   index=False)

print(f"Train samples : {len(train_df):,}  →  {train_csv}")
print(f"Val   samples : {len(val_df):,}  →  {val_csv}")
print("\nSample rows:")
print(train_df.head(3).to_string(index=False))
print("\n✓ Data preparation complete.")
