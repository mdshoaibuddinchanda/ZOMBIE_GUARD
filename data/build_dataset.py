import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import extract_features

MALICIOUS_DIR = "data/raw/malicious"
BENIGN_DIR = "data/raw/benign"
OUTPUT_FEATURES = "data/processed/features.csv"
OUTPUT_LABELS = "data/processed/labels.csv"

os.makedirs("data/processed", exist_ok=True)


def process_directory(directory: str, label: int) -> list:
    rows = []
    files = [name for name in os.listdir(directory) if name.endswith(".zip")]
    print(f"\nProcessing {len(files)} files from {directory} (label={label})")

    for i, filename in enumerate(files):
        file_path = os.path.join(directory, filename)
        features = extract_features(file_path)
        features["filename"] = filename
        features["label"] = label
        rows.append(features)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} done")

    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += process_directory(MALICIOUS_DIR, label=1)
    all_rows += process_directory(BENIGN_DIR, label=0)

    df = pd.DataFrame(all_rows)

    feature_cols = [
        "lf_compression_method",
        "cd_compression_method",
        "method_mismatch",
        "data_entropy_shannon",
        "data_entropy_renyi",
        "declared_vs_entropy_flag",
        "eocd_count",
        "lf_unknown_method",
        "file_size_bytes",
    ]

    print(f"\nDataset shape: {df.shape}")
    print(f"Malicious: {(df['label'] == 1).sum()}")
    print(f"Benign:    {(df['label'] == 0).sum()}")
    print(f"NaN values: {df[feature_cols].isna().sum().sum()}")

    # Save features and labels separately
    df[["filename"] + feature_cols].to_csv(OUTPUT_FEATURES, index=False)
    df[["filename", "label"]].to_csv(OUTPUT_LABELS, index=False)

    print(f"\nSaved features to: {OUTPUT_FEATURES}")
    print(f"Saved labels to:   {OUTPUT_LABELS}")