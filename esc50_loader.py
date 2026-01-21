import os
import pandas as pd

# Base ESC-50 path (relative to project root)
DATASET_PATH = "data/ESC50/"
META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")

def load_esc50_metadata(selected_folds=None):
    df = pd.read_csv(META_FILE)
    # Create full file path for each audio file
    df["file_path"] = df["filename"].apply(
        lambda name: os.path.join(AUDIO_DIR, name)
    )
    # Optionally use only some folds (1–5)
    if selected_folds is not None:
        df = df[df["fold"].isin(selected_folds)]
    return df[["file_path", "target", "category"]]

if __name__ == "__main__":
    meta = load_esc50_metadata()
    print(meta.head())
    print("Total files:", len(meta))
