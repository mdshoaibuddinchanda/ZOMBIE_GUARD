import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.extractor import extract_features

TEST_EV = "data/real_splits/test/evasion"
TEST_NEV = "data/real_splits/test/non_evasion"

rows = []
for fname in os.listdir(TEST_EV):
    if fname.endswith(".zip"):
        feat = extract_features(os.path.join(TEST_EV, fname))
        feat["label"] = 1
        feat["file"] = fname
        rows.append(feat)

for fname in os.listdir(TEST_NEV):
    if fname.endswith(".zip"):
        feat = extract_features(os.path.join(TEST_NEV, fname))
        feat["label"] = 0
        feat["file"] = fname
        rows.append(feat)

df = pd.DataFrame(rows)
ev = df[df["label"] == 1]
nev = df[df["label"] == 0]

print("EVASION TEST SAMPLES:")
print(f"  eocd_count mean:      {ev.eocd_count.mean():.2f}")
print(f"  method_mismatch mean: {ev.method_mismatch.mean():.4f}")
print(f"  entropy mean:         {ev.data_entropy_shannon.mean():.4f}")
print(f"  suspicious_ratio mean:{ev.suspicious_entry_ratio.mean():.4f}")

print("NON-EVASION TEST SAMPLES:")
print(f"  eocd_count mean:      {nev.eocd_count.mean():.2f}")
print(f"  method_mismatch mean: {nev.method_mismatch.mean():.4f}")
print(f"  entropy mean:         {nev.data_entropy_shannon.mean():.4f}")
print(f"  suspicious_ratio mean:{nev.suspicious_entry_ratio.mean():.4f}")
