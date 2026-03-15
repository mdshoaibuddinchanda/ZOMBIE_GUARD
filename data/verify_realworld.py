import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import extract_features
from src.classifier import load_model, predict

VALIDATION_DIR = "data/real_world_validation"
MODEL_PATH = "models/xgboost_model.pkl"

model = load_model(MODEL_PATH)

detected = []
clean = []
errors = []

files = [f for f in os.listdir(VALIDATION_DIR) if f.endswith(".zip")]

for fname in sorted(files):
    fpath = os.path.join(VALIDATION_DIR, fname)
    try:
        features = extract_features(fpath)
        result = predict(model, features)

        entry = {
            "file": fname,
            "probability": result["probability"],
            "lf_method": features["lf_compression_method"],
            "cd_method": features["cd_compression_method"],
            "mismatch": features["method_mismatch"],
            "entropy": round(features["data_entropy_shannon"], 3),
            "entropy_flag": features["declared_vs_entropy_flag"],
            "eocd_count": features["eocd_count"],
        }

        if result["label"] == 1:
            detected.append(entry)
        else:
            clean.append(entry)

    except Exception as e:
        errors.append((fname, str(e)))

print(f"\n-- DETECTED ({len(detected)} files) --------------------------")
print(
    f"{'File':<25} {'Prob':>6} {'LFH':>4} {'CDH':>4} "
    f"{'Mismatch':>9} {'Entropy':>8} {'EOCD':>5}"
)
print("-" * 75)

mismatch_count = 0
entropy_count = 0
gootloader_count = 0

for e in detected:
    print(
        f"{e['file'][:24]:<25} {e['probability']:>6.1%} "
        f"{e['lf_method']:>4} {e['cd_method']:>4} "
        f"{str(e['mismatch']):>9} {e['entropy']:>8.3f} "
        f"{e['eocd_count']:>5}"
    )
    if e["mismatch"]:
        mismatch_count += 1
    if e["entropy_flag"]:
        entropy_count += 1
    if e["eocd_count"] > 1:
        gootloader_count += 1

print("\n-- DETECTION SIGNAL BREAKDOWN -------------------------------")
print(f"  Method mismatch (LFH!=CDH)    : {mismatch_count} files")
print(f"  Entropy mismatch flag         : {entropy_count} files")
print(f"  Gootloader-style (EOCD > 1)   : {gootloader_count} files")
print(f"  Total detected                : {len(detected)} files")

print("\n-- CLEAN (not header evasion) -------------------------------")
print(f"  Files with correct headers    : {len(clean)}")
print("  (These carry malware via other delivery mechanisms)")

print("\n-- ERRORS ---------------------------------------------------")
print(f"  Parse errors                  : {len(errors)}")
for fname, err in errors[:5]:
    print(f"  {fname}: {err}")

print("\n-- SUMMARY --------------------------------------------------")
print(f"  Total scanned    : {len(files)}")
print(f"  Detected         : {len(detected)} ({len(detected)/len(files)*100:.1f}%)")
print(f"  Clean            : {len(clean)} ({len(clean)/len(files)*100:.1f}%)")
print(f"  Errors           : {len(errors)}")
print("------------------------------------------------------------")

output_csv = "data/realworld_labels.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "signal"])
    for e in detected:
        signal = (
            "gootloader"
            if e["eocd_count"] > 1
            else "mismatch"
            if e["mismatch"]
            else "entropy"
            if e["entropy"] > 7.5
            else "other"
        )
        writer.writerow([e["file"], 1, signal])
    for e in clean:
        writer.writerow([e["file"], 0, "none"])

print(f"\nLabels saved to: {output_csv}")
