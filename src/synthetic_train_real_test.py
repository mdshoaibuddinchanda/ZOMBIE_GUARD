"""Train on synthetic-only data and evaluate strictly on real-world samples."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import FEATURE_COLS
from src.realworld_features import load_realworld_features

SYNTH_FEATURES = Path("data/processed/features.csv")
SYNTH_LABELS = Path("data/processed/labels.csv")
OUT_CSV = Path("paper/figures/csv/table_synthetic_train_real_test.csv")


def load_synthetic() -> tuple[pd.DataFrame, pd.Series]:
    """Load synthetic train matrix and labels."""
    features_df = pd.read_csv(SYNTH_FEATURES)
    labels_df = pd.read_csv(SYNTH_LABELS)
    merged = features_df.merge(labels_df, on="filename", how="inner")

    for col in FEATURE_COLS:
        if col not in merged.columns:
            merged[col] = 0

    x_train = merged[FEATURE_COLS].astype(float)
    y_train = merged["label"].astype(int)
    return x_train, y_train


def build_model() -> LGBMClassifier:
    """Build baseline LightGBM model used in paper experiments."""
    return LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    """Run strict synthetic-to-real transfer experiment."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Synthetic-Train + Real-Test Experiment")
    logger.info("Loading synthetic training data...")
    x_train, y_train = load_synthetic()
    logger.info(f"  ✓ Loaded {len(x_train)} synthetic samples")
    
    logger.info("Loading real-world test data...")
    real_df = load_realworld_features(refresh=False)
    logger.info(f"  ✓ Loaded {len(real_df)} real-world samples")

    x_test = real_df[FEATURE_COLS].astype(float)
    y_test = real_df["label"].astype(int)

    logger.info("Training LightGBM model...")
    model = build_model()
    model.fit(x_train, y_train)
    logger.info(f"  ✓ Model trained")

    logger.info("Evaluating on real-world test set...")
    y_pred = np.asarray(model.predict(x_test)).astype(int).ravel()
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    logger.info(f"  ✓ Predictions complete")

    result_df = pd.DataFrame(
        [
            {
                "train_dataset": "Synthetic only",
                "test_dataset": "Real world (1,366)",
                "precision": float(precision),
                "recall": float(recall),
                "false_positive_rate": float(fp_rate),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        ]
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    logger.info(f"\n✓ Results saved to {OUT_CSV}")
    logger.info(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.2%} | Recall: {recall:.2%} | FP-Rate: {fp_rate:.2%}")


if __name__ == "__main__":
    main()
