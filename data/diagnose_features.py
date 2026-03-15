import numpy as np
import pandas as pd

features_df = pd.read_csv("data/processed/features.csv")
labels_df = pd.read_csv("data/processed/labels.csv")
merged = features_df.merge(labels_df, on="filename", how="inner")

mal = merged[merged["label"] == 1]
ben = merged[merged["label"] == 0]

check_cols = [
    "entry_count",
    "suspicious_entry_count",
    "suspicious_entry_ratio",
    "entropy_variance",
    "lf_crc_valid",
    "any_crc_mismatch",
    "is_encrypted",
]

print("-- Feature distribution: Malicious vs Benign --")
print(f"{'Feature':<30} {'Mal mean':>10} {'Ben mean':>10} {'Mal std':>10}")
print("-" * 65)
for col in check_cols:
    if col in merged.columns:
        print(
            f"{col:<30} "
            f"{mal[col].mean():>10.4f} "
            f"{ben[col].mean():>10.4f} "
            f"{mal[col].std():>10.4f}"
        )

print("\n-- Entry count distribution --")
print("Malicious entry_count value counts (top 10):")
print(mal["entry_count"].value_counts().head(10))
print("\nBenign entry_count value counts (top 10):")
print(ben["entry_count"].value_counts().head(10))
