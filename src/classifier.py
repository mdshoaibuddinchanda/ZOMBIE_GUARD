"""
classifier.py
XGBoost training, evaluation, and inference utilities for ZombieGuard features.
Part of ZombieGuard - Archive Header Evasion Detection System.
CVE-2026-0866 | https://github.com/YOUR_USERNAME/zombieguard
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# ── Paths ────────────────────────────────────────────
FEATURES_PATH = "data/processed/features.csv"
LABELS_PATH   = "data/processed/labels.csv"
MODEL_PATH    = "models/xgboost_model.pkl"

FEATURE_COLS = [
    # Core attack signals - format agnostic
    "lf_compression_method",
    "cd_compression_method",
    "method_mismatch",
    "data_entropy_shannon",
    "data_entropy_renyi",
    "declared_vs_entropy_flag",
    "eocd_count",
    "lf_unknown_method",
    # Per-entry inconsistency signals only
    "suspicious_entry_count",
    "suspicious_entry_ratio",
    "any_crc_mismatch",
    # Encryption signal
    "is_encrypted",
]


# ── Config ───────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Configuration container for model training and validation settings."""
    random_state: int = 42
    n_splits: int = 5
    test_size: float = 0.2


# ── Model builder ─────────────────────────────────────
def _build_model(random_state: int) -> XGBClassifier:
    """Construct the configured XGBoost binary classifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


# ── Metrics ───────────────────────────────────────────
def _compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    }


# ── Training ──────────────────────────────────────────
def train_with_cross_validation(
    features_path: str = FEATURES_PATH,
    labels_path: str   = LABELS_PATH,
    config: TrainingConfig | None = None,
):
    """Train and evaluate the XGBoost model with CV plus holdout testing."""
    if config is None:
        config = TrainingConfig()

    # Load and merge
    features_df = pd.read_csv(features_path)
    labels_df   = pd.read_csv(labels_path)
    merged = features_df.merge(labels_df, on="filename", how="inner")

    # Convert boolean columns to int (XGBoost requirement)
    for col in [
        "method_mismatch",
        "declared_vs_entropy_flag",
        "lf_crc_valid",
        "any_crc_mismatch",
        "is_encrypted",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)

    X = merged[FEATURE_COLS]
    y = merged["label"]

    print(f"Dataset loaded: {len(merged)} samples")
    print(f"  Malicious : {(y == 1).sum()}")
    print(f"  Benign    : {(y == 0).sum()}")
    print(f"  Features  : {list(X.columns)}\n")

    # Holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # ── 5-fold cross-validation on training set
    print(f"Running {config.n_splits}-fold stratified cross-validation...")
    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state
    )
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        model = _build_model(config.random_state + fold_idx)
        model.fit(X_fold_train, y_fold_train)

        val_pred = model.predict(X_fold_val)
        val_prob = model.predict_proba(X_fold_val)[:, 1]
        metrics = _compute_metrics(y_fold_val, val_pred, val_prob)
        metrics["fold"] = float(fold_idx)
        fold_metrics.append(metrics)

        print(f"  Fold {fold_idx}: "
              f"acc={metrics['accuracy']:.4f}  "
              f"rec={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"auc={metrics['roc_auc']:.4f}")

    cv_df = pd.DataFrame(fold_metrics)
    cv_summary = cv_df.drop(columns=["fold"]).mean().to_dict()

    print(f"\nCV mean — "
          f"acc={cv_summary['accuracy']:.4f}  "
          f"rec={cv_summary['recall']:.4f}  "
          f"f1={cv_summary['f1']:.4f}  "
          f"auc={cv_summary['roc_auc']:.4f}")

    # ── Final model trained on full training set
    print("\nTraining final model on full training set...")
    final_model = _build_model(config.random_state)
    final_model.fit(X_train, y_train)

    # ── Holdout test evaluation
    test_pred = final_model.predict(X_test)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_pred, test_prob)

    print("\n── Holdout Test Set Results ─────────────────────────")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print("\n── Classification Report ────────────────────────────")
    print(classification_report(y_test, test_pred,
          target_names=["Benign", "Malicious"]))
    print("── Confusion Matrix ─────────────────────────────────")
    cm = confusion_matrix(y_test, test_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("─────────────────────────────────────────────────────")

    return {
        "cv_fold_metrics": cv_df,
        "cv_summary":      cv_summary,
        "test_metrics":    test_metrics,
        "model":           final_model,
        "feature_cols":    FEATURE_COLS,
        "X_test":          X_test,
        "y_test":          y_test,
        "test_pred":       test_pred,
        "test_prob":       test_prob,
    }


# ── Save / Load ───────────────────────────────────────
def save_model(model: XGBClassifier, path: str = MODEL_PATH):
    """Persist a trained XGBoost model to disk."""
    os.makedirs(Path(path).parent, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load a persisted XGBoost model from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No model found at: {path}")
    return joblib.load(path)


# ── Predict (used by detector.py) ────────────────────
def predict(model: XGBClassifier, features: dict) -> dict:
    """
    Takes a feature dictionary from extractor.extract_features()
    and returns a prediction verdict.

    Returns:
        {
          "label":       1 or 0,
          "verdict":     "ZOMBIE ZIP DETECTED" or "CLEAN",
          "probability": float (0.0 to 1.0),
        }
    """
    # Build single-row dataframe in correct column order
    row = {}
    for col in FEATURE_COLS:
        val = features.get(col, 0)
        # Convert booleans to int
        row[col] = int(val) if isinstance(val, bool) else val

    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[0][1])

    # Safety override for clear parser-confusion signatures.
    # This keeps model-led behavior but ensures known high-risk structural
    # contradictions are not downgraded by dataset drift.
    strong_structural_evasion = (
        bool(features.get("method_mismatch", False))
        and bool(features.get("declared_vs_entropy_flag", False))
        and float(features.get("suspicious_entry_ratio", 0.0)) >= 0.5
    )
    if strong_structural_evasion:
        prob = max(prob, 0.95)

    label = int(prob >= 0.5)

    return {
        "label":       label,
        "verdict":     "ZOMBIE ZIP DETECTED" if label == 1 else "CLEAN",
        "probability": round(prob, 4),
    }


# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    results = train_with_cross_validation()
    save_model(results["model"])
