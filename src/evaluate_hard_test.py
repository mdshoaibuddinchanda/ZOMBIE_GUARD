"""
evaluate_hard_test.py
Evaluates all three models on the hard test set.
Hard test set forces multi-feature detection -
eocd_count alone cannot separate classes (ratio 1.18x).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
from src.extractor import extract_features

HARD_EV_DIR = "data/hard_test/evasion"
HARD_NEV_DIR = "data/hard_test/non_evasion"

MODEL_A = "models/xgboost_model.pkl"
MODEL_B = "models/xgboost_real.pkl"
MODEL_C = "models/xgboost_mixed.pkl"

FEATURE_COLS = [
    "lf_compression_method",
    "cd_compression_method",
    "method_mismatch",
    "data_entropy_shannon",
    "data_entropy_renyi",
    "declared_vs_entropy_flag",
    "eocd_count",
    "lf_unknown_method",
    "suspicious_entry_count",
    "suspicious_entry_ratio",
    "any_crc_mismatch",
    "is_encrypted",
]


def build_test_df() -> tuple:
    """Extract features from hard test set."""
    rows = []

    print("Extracting features from hard test set...")
    ev_files = list(Path(HARD_EV_DIR).glob("*.zip"))
    nev_files = list(Path(HARD_NEV_DIR).glob("*.zip"))

    print(f"  Evasion    : {len(ev_files)}")
    print(f"  Non-evasion: {len(nev_files)}")

    for fpath in ev_files:
        f = extract_features(str(fpath))
        f["filename"] = fpath.name
        f["label"] = 1
        rows.append(f)

    for fpath in nev_files:
        f = extract_features(str(fpath))
        f["filename"] = fpath.name
        f["label"] = 0
        rows.append(f)

    df = pd.DataFrame(rows)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)

    return X, y, df


def evaluate_model(model, X, y, name: str) -> dict:
    """Evaluate one model and print full results."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(
            y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(
            y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(
            y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "TP": int(cm[1, 1]), "FP": int(cm[0, 1]),
        "TN": int(cm[0, 0]), "FN": int(cm[1, 0]),
    }

    print(f"\n-- {name} --------------------------------------")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}  "
          f"FN={cm[1, 0]}  TP={cm[1, 1]}")
    print(f"\n{classification_report(y, y_pred, target_names=['Non-evasion', 'Evasion'])}")

    # Show which files were missed
    missed_idx = [i for i, (t, p) in enumerate(
        zip(y, y_pred)) if t == 1 and p == 0
    ]
    if missed_idx:
        print(f"  Missed evasion files ({len(missed_idx)}):")
        for i in missed_idx[:5]:
            print(f"    - row {i}")

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("ZombieGuard - Hard Test Set Evaluation")
    print("EOCD ratio: 1.18x (multi-feature required)")
    print("=" * 55)

    X_test, y_test, df = build_test_df()
    print(f"\nTest set: {len(X_test)} samples "
          f"(evasion={y_test.sum()}, "
          f"non-evasion={(y_test == 0).sum()})")

    all_results = []

    # Model A - Synthetic trained
    print("\n" + "=" * 55)
    model_a = joblib.load(MODEL_A)
    r = evaluate_model(
        model_a, X_test, y_test,
        "Model A - Synthetic-trained"
    )
    all_results.append(r)

    # Model B - Real trained
    print("\n" + "=" * 55)
    model_b = joblib.load(MODEL_B)
    r = evaluate_model(
        model_b, X_test, y_test,
        "Model B - Real-trained"
    )
    all_results.append(r)

    # Model C - Mixed trained
    print("\n" + "=" * 55)
    model_c = joblib.load(MODEL_C)
    r = evaluate_model(
        model_c, X_test, y_test,
        "Model C - Mixed"
    )
    all_results.append(r)

    # Final comparison
    print("\n" + "=" * 55)
    print("FINAL COMPARISON - Hard test set")
    print("(EOCD ratio 1.18x - multi-feature detection required)")
    print("=" * 55)

    results_df = pd.DataFrame(all_results)
    print(results_df[[
        "model", "recall", "precision",
        "f1", "roc_auc", "FP", "FN"
    ]].to_string(index=False))

    os.makedirs("paper/figures", exist_ok=True)
    results_df.to_csv(
        "paper/figures/hard_test_comparison.csv",
        index=False
    )
    print("\nSaved: paper/figures/hard_test_comparison.csv")
