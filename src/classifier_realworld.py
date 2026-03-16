"""
classifier_realworld.py
ZombieGuard - Three-model comparison experiment.

Compares:
  Model A: Synthetic-trained (existing model)
  Model B: Real-data-trained (65 evasion + 890 non-evasion)
  Model C: Mixed-trained (real + synthetic combined)

All three evaluated on same held-out real test set.
This answers: does synthetic training generalise to real attacks?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import extract_features

# -- Paths -------------------------------------------------------
REAL_SPLITS = "data/real_splits"
SYNTH_FEAT = "data/processed/features.csv"
SYNTH_LABELS = "data/processed/labels.csv"
MODEL_A_PATH = "models/xgboost_model.pkl"
MODEL_B_PATH = "models/xgboost_real.pkl"
MODEL_C_PATH = "models/xgboost_mixed.pkl"
RESULTS_PATH = "paper/figures/three_model_comparison.csv"

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


# -- Feature extraction ------------------------------------------
def extract_from_directory(directory: str, label: int) -> pd.DataFrame:
    """Extract features from all ZIP files in a directory."""
    rows = []
    files = list(Path(directory).glob("*.zip"))
    print(f"  Extracting {len(files)} files from {directory} (label={label})")

    for i, fpath in enumerate(files):
        features = extract_features(str(fpath))
        features["filename"] = fpath.name
        features["label"] = label
        rows.append(features)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(files)} done")

    return pd.DataFrame(rows)


def build_real_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train, val, test DataFrames from real splits."""
    print("\nBuilding real dataset...")

    # Train
    tr_ev = extract_from_directory(f"{REAL_SPLITS}/train/evasion", 1)
    tr_nev = extract_from_directory(f"{REAL_SPLITS}/train/non_evasion", 0)
    train_df = pd.concat([tr_ev, tr_nev], ignore_index=True)

    # Validation
    val_ev = extract_from_directory(f"{REAL_SPLITS}/val/evasion", 1)
    val_nev = extract_from_directory(f"{REAL_SPLITS}/val/non_evasion", 0)
    val_df = pd.concat([val_ev, val_nev], ignore_index=True)

    # Test - held out completely
    te_ev = extract_from_directory(f"{REAL_SPLITS}/test/evasion", 1)
    te_nev = extract_from_directory(f"{REAL_SPLITS}/test/non_evasion", 0)
    test_df = pd.concat([te_ev, te_nev], ignore_index=True)

    print(f"\n  Train : {len(train_df)} (evasion={len(tr_ev)}, non-evasion={len(tr_nev)})")
    print(f"  Val   : {len(val_df)} (evasion={len(val_ev)}, non-evasion={len(val_nev)})")
    print(f"  Test  : {len(test_df)} (evasion={len(te_ev)}, non-evasion={len(te_nev)})")

    return train_df, val_df, test_df


def build_synthetic_dataset() -> pd.DataFrame:
    """Load synthetic training data."""
    feat_df = pd.read_csv(SYNTH_FEAT)
    labels_df = pd.read_csv(SYNTH_LABELS)
    merged = feat_df.merge(labels_df, on="filename")

    for col in [
        "method_mismatch",
        "declared_vs_entropy_flag",
        "lf_unknown_method",
        "any_crc_mismatch",
        "is_encrypted",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    return merged


# -- Model training ----------------------------------------------
def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and labels from DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    x = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)
    return x, y


def evaluate(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Evaluate model and print results."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n-- {model_name} Results --------------------------")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")
    print(
        "\n"
        + classification_report(y_test, y_pred, target_names=["Non-evasion", "Evasion"])
    )

    return metrics


def cross_val_with_ci(
    model_factory,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """5-fold CV with 95% confidence intervals."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"recall": [], "f1": [], "roc_auc": []}

    for fold, (tr, val) in enumerate(skf.split(x, y), start=1):
        _ = fold
        model = model_factory()
        model.fit(x.iloc[tr], y.iloc[tr])
        y_pred = model.predict(x.iloc[val])
        y_prob = model.predict_proba(x.iloc[val])[:, 1]
        scores["recall"].append(recall_score(y.iloc[val], y_pred, zero_division=0))
        scores["f1"].append(f1_score(y.iloc[val], y_pred, zero_division=0))
        scores["roc_auc"].append(roc_auc_score(y.iloc[val], y_prob))

    ci_results = {}
    for metric, vals in scores.items():
        mean = np.mean(vals)
        std = np.std(vals)
        ci = stats.t.interval(0.95, df=len(vals) - 1, loc=mean, scale=stats.sem(vals))
        ci_results[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_low": round(ci[0], 4),
            "ci_high": round(ci[1], 4),
        }

    return ci_results


# -- Main experiment ---------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("ZombieGuard - Three-Model Comparison Experiment")
    print("=" * 55)

    # Build datasets
    train_real, val_real, test_real = build_real_dataset()
    synth_df = build_synthetic_dataset()

    # Common test set - same for all three models
    X_test, y_test = prepare_xy(test_real)
    print(
        f"\nHeld-out test set: {len(X_test)} samples "
        f"(evasion={y_test.sum()}, non-evasion={(y_test == 0).sum()})"
    )

    all_results = []

    # MODEL A - Synthetic-trained (load existing)
    print("\n" + "=" * 55)
    print("MODEL A - Synthetic-trained (existing model)")
    print("=" * 55)

    model_a = joblib.load(MODEL_A_PATH)
    metrics_a = evaluate(model_a, X_test, y_test, "Model A (Synthetic)")
    all_results.append(metrics_a)

    # MODEL B - Real-data-trained
    print("\n" + "=" * 55)
    print("MODEL B - Real-data-trained")
    print(f"  Train: {len(train_real)} samples")
    print("=" * 55)

    X_b_train, y_b_train = prepare_xy(train_real)
    X_b_val, y_b_val = prepare_xy(val_real)
    _ = (X_b_val, y_b_val)

    print("\n5-fold CV on real training data...")
    ci_b = cross_val_with_ci(build_xgb, X_b_train, y_b_train)
    print(
        f"  Recall  : {ci_b['recall']['mean']:.4f} "
        f"+- {ci_b['recall']['std']:.4f} "
        f"(95% CI: {ci_b['recall']['ci_low']:.4f}-{ci_b['recall']['ci_high']:.4f})"
    )
    print(f"  F1      : {ci_b['f1']['mean']:.4f} +- {ci_b['f1']['std']:.4f}")
    print(f"  ROC-AUC : {ci_b['roc_auc']['mean']:.4f} +- {ci_b['roc_auc']['std']:.4f}")

    model_b = build_xgb()
    model_b.fit(X_b_train, y_b_train)
    joblib.dump(model_b, MODEL_B_PATH)

    metrics_b = evaluate(model_b, X_test, y_test, "Model B (Real only)")
    metrics_b["cv_recall"] = ci_b["recall"]["mean"]
    metrics_b["cv_recall_ci"] = f"{ci_b['recall']['ci_low']:.4f}-{ci_b['recall']['ci_high']:.4f}"
    all_results.append(metrics_b)

    # MODEL C - Mixed (real + synthetic)
    print("\n" + "=" * 55)
    print("MODEL C - Mixed (real + synthetic)")
    print("=" * 55)

    # Combine real train with subset of synthetic
    synth_mal = synth_df[synth_df["label"] == 1].sample(500, random_state=42)
    synth_ben = synth_df[synth_df["label"] == 0].sample(500, random_state=42)
    synth_subset = pd.concat([synth_mal, synth_ben], ignore_index=True)

    mixed_df = pd.concat([train_real, synth_subset], ignore_index=True)
    X_c_train, y_c_train = prepare_xy(mixed_df)
    print(
        f"  Mixed train size: {len(mixed_df)} "
        f"(real={len(train_real)}, synthetic={len(synth_subset)})"
    )

    print("\n5-fold CV on mixed training data...")
    ci_c = cross_val_with_ci(build_xgb, X_c_train, y_c_train)
    print(
        f"  Recall  : {ci_c['recall']['mean']:.4f} "
        f"+- {ci_c['recall']['std']:.4f} "
        f"(95% CI: {ci_c['recall']['ci_low']:.4f}-{ci_c['recall']['ci_high']:.4f})"
    )
    print(f"  F1      : {ci_c['f1']['mean']:.4f} +- {ci_c['f1']['std']:.4f}")
    print(f"  ROC-AUC : {ci_c['roc_auc']['mean']:.4f} +- {ci_c['roc_auc']['std']:.4f}")

    model_c = build_xgb()
    model_c.fit(X_c_train, y_c_train)
    joblib.dump(model_c, MODEL_C_PATH)

    metrics_c = evaluate(model_c, X_test, y_test, "Model C (Mixed)")
    metrics_c["cv_recall"] = ci_c["recall"]["mean"]
    metrics_c["cv_recall_ci"] = f"{ci_c['recall']['ci_low']:.4f}-{ci_c['recall']['ci_high']:.4f}"
    all_results.append(metrics_c)

    # Final comparison table
    print("\n" + "=" * 55)
    print("FINAL COMPARISON - Same held-out real test set")
    print("=" * 55)

    results_df = pd.DataFrame(all_results)
    print(results_df[["model", "recall", "f1", "roc_auc", "precision"]].to_string(index=False))

    os.makedirs("paper/figures", exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved: {RESULTS_PATH}")

    print("\n" + "=" * 55)
    print("Three-model experiment complete.")
    print("Models saved:")
    print(f"  {MODEL_A_PATH}")
    print(f"  {MODEL_B_PATH}")
    print(f"  {MODEL_C_PATH}")
