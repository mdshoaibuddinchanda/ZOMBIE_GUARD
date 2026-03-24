import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS_CSV = "data/realworld_labels.csv"
VAL_DIR = "data/real_world_validation"

# Three-way split directories
DIRS = {
    "train": {
        "evasion": "data/real_splits/train/evasion",
        "non_evasion": "data/real_splits/train/non_evasion",
    },
    "val": {
        "evasion": "data/real_splits/val/evasion",
        "non_evasion": "data/real_splits/val/non_evasion",
    },
    "test": {
        "evasion": "data/real_splits/test/evasion",
        "non_evasion": "data/real_splits/test/non_evasion",
    },
}

for split in DIRS.values():
    for path in split.values():
        os.makedirs(path, exist_ok=True)


def copy_files(file_list: list, src_dir: str, dst_dir: str) -> int:
    copied = 0
    for fname in file_list:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
    return copied


df = pd.read_csv(LABELS_CSV)

evasion = df[df["label"] == 1].reset_index(drop=True)
non_ev = df[df["label"] == 0].reset_index(drop=True)

print(f"Total evasion positives : {len(evasion)}")
print(f"Total non-evasion       : {len(non_ev)}")

# Split evasion: 70% train, 15% val, 15% test
ev_train, ev_temp = train_test_split(
    evasion, test_size=0.30, random_state=42, shuffle=True
)
ev_val, ev_test = train_test_split(
    ev_temp, test_size=0.50, random_state=42, shuffle=True
)

# Split non-evasion: same ratios
nev_train, nev_temp = train_test_split(
    non_ev, test_size=0.30, random_state=42, shuffle=True
)
nev_val, nev_test = train_test_split(
    nev_temp, test_size=0.50, random_state=42, shuffle=True
)

# Copy files
c = {}
c["ev_train"] = copy_files(ev_train["filename"].tolist(), VAL_DIR, DIRS["train"]["evasion"])
c["ev_val"] = copy_files(ev_val["filename"].tolist(), VAL_DIR, DIRS["val"]["evasion"])
c["ev_test"] = copy_files(ev_test["filename"].tolist(), VAL_DIR, DIRS["test"]["evasion"])
c["nev_train"] = copy_files(
    nev_train["filename"].tolist(), VAL_DIR, DIRS["train"]["non_evasion"]
)
c["nev_val"] = copy_files(nev_val["filename"].tolist(), VAL_DIR, DIRS["val"]["non_evasion"])
c["nev_test"] = copy_files(
    nev_test["filename"].tolist(), VAL_DIR, DIRS["test"]["non_evasion"]
)

# Report
print("\n-- Split Summary --------------------------------")
print(f"{'Split':<12} {'Evasion':>10} {'Non-evasion':>12} {'Total':>8}")
print(f"{'-' * 46}")
print(f"{'Train':<12} {c['ev_train']:>10} {c['nev_train']:>12} {c['ev_train'] + c['nev_train']:>8}")
print(f"{'Validation':<12} {c['ev_val']:>10} {c['nev_val']:>12} {c['ev_val'] + c['nev_val']:>8}")
print(f"{'Test':<12} {c['ev_test']:>10} {c['nev_test']:>12} {c['ev_test'] + c['nev_test']:>8}")
print(f"{'-' * 46}")
total_ev = c['ev_train'] + c['ev_val'] + c['ev_test']
total_nev = c['nev_train'] + c['nev_val'] + c['nev_test']
print(f"{'Total':<12} {total_ev:>10} {total_nev:>12} {total_ev + total_nev:>8}")
print("\nSaved to: data/real_splits/")

# Save split labels for reference
splits_df = pd.concat(
    [
        ev_train.assign(split='train'),
        ev_val.assign(split='val'),
        ev_test.assign(split='test'),
        nev_train.assign(split='train'),
        nev_val.assign(split='val'),
        nev_test.assign(split='test'),
    ]
)
splits_df.to_csv("data/realworld_splits.csv", index=False)
print("Split labels: data/realworld_splits.csv")
