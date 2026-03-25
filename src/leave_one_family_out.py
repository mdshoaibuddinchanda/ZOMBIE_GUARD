"""Leave-one-family-out validation to test family-specific memorization risk."""

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
OUT_CSV = Path("paper/figures/csv/table_leave_one_family_out.csv")


def load_synthetic_df() -> pd.DataFrame:
    """Load synthetic features with labels for baseline training support."""
    feat_df = pd.read_csv(SYNTH_FEATURES)
    label_df = pd.read_csv(SYNTH_LABELS)
    synth_df = feat_df.merge(label_df, on="filename", how="inner")
    for col in FEATURE_COLS:
        if col not in synth_df.columns:
            synth_df[col] = 0
    synth_df["family"] = "Synthetic"
    return synth_df


def build_model() -> LGBMClassifier:
    """Create the leave-one-family-out evaluation model."""
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


def family_candidates(real_df: pd.DataFrame, min_positives: int) -> list[str]:
    """Return families with enough positive samples for stable recall estimates."""
    positive = real_df[real_df["label"] == 1].copy()
    counts = positive.groupby("family").size().sort_values(ascending=False)
    return [str(family) for family, count in counts.items() if count >= min_positives]


def main() -> None:
    """Run leave-one-family-out experiment and save per-family metrics."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Leave-One-Family-Out Validation")
    logger.info("Loading synthetic and real-world datasets...")
    real_df = load_realworld_features(refresh=False)
    synth_df = load_synthetic_df()
    logger.info(f"  ✓ Loaded {len(synth_df)} synthetic + {len(real_df)} real samples")

    families = family_candidates(real_df, min_positives=5)
    if not families:
        raise RuntimeError("No families with enough positives for leave-one-family-out test.")
    logger.info(f"  ✓ Found {len(families)} families with ≥5 positive samples")

    rows: list[dict[str, float | int | str]] = []

    for idx, family in enumerate(families, start=1):
        logger.info(f"\n[{idx}/{len(families)}] Processing family: {family}")
        holdout_pos = real_df[(real_df["label"] == 1) & (real_df["family"] == family)].copy()
        holdout_neg = real_df[real_df["label"] == 0].copy()

        train_real = real_df[~((real_df["label"] == 1) & (real_df["family"] == family))].copy()
        train_df = pd.concat([synth_df, train_real], ignore_index=True)

        x_train = train_df[FEATURE_COLS].astype(float)
        y_train = train_df["label"].astype(int)

        test_df = pd.concat([holdout_pos, holdout_neg], ignore_index=True)
        x_test = test_df[FEATURE_COLS].astype(float)
        y_test = test_df["label"].astype(int)

        logger.info(f"  Training on {len(train_df)} samples...")
        model = build_model()
        model.fit(x_train, y_train)
        y_pred = np.asarray(model.predict(x_test)).astype(int).ravel()

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
        logger.info(f"  → Recall: {recall:.2%} | Precision: {precision:.2%} | FP-Rate: {fp_rate:.2%}")

        rows.append(
            {
                "held_out_family": family,
                "train_rows": int(len(train_df)),
                "test_positive_rows": int(len(holdout_pos)),
                "test_negative_rows": int(len(holdout_neg)),
                "recall": float(recall),
                "precision": float(precision),
                "false_positive_rate": float(fp_rate),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("test_positive_rows", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    logger.info(f"\n✓ Leave-one-family-out complete. Results saved to {OUT_CSV}")
    logger.info(f"Mean Recall: {pd.to_numeric(result_df['recall']).mean():.2%}")


if __name__ == "__main__":
    main()
