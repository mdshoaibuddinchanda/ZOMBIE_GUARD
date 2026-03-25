"""Feature-group ablation on real-only train/test split."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import FEATURE_COLS
from src.realworld_features import load_realworld_features

OUT_CSV = Path("paper/figures/csv/table_real_only_ablation.csv")

FEATURE_GROUPS: dict[str, list[str]] = {
    "Entropy features": [
        "data_entropy_shannon",
        "data_entropy_renyi",
        "declared_vs_entropy_flag",
    ],
    "Suspicious entry group": [
        "method_mismatch",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
    ],
    "Header method fields": [
        "lf_compression_method",
        "cd_compression_method",
    ],
    "CRC mismatch": ["any_crc_mismatch"],
    "EOCD count": ["eocd_count"],
    "Encryption flag": ["is_encrypted"],
}


def build_model() -> LGBMClassifier:
    """Build LightGBM model for ablation runs."""
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
    """Run real-only ablation and save recall deltas."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Real-Only Feature Ablation Study")
    logger.info("Loading real-world dataset...")
    real_df = load_realworld_features(refresh=False)
    logger.info(f"  ✓ Loaded {len(real_df)} samples")

    x = real_df[FEATURE_COLS].astype(float)
    y = real_df["label"].astype(int)

    logger.info("Splitting: 80% train, 20% test...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info(f"  ✓ Train: {len(x_train)}, Test: {len(x_test)}")

    logger.info("Training baseline model...")
    full_model = build_model()
    full_model.fit(x_train, y_train)
    base_pred = np.asarray(full_model.predict(x_test)).astype(int).ravel()
    base_recall = recall_score(y_test, base_pred, zero_division=0)
    logger.info(f"  ✓ Baseline recall: {base_recall:.2%}")

    logger.info("Ablating feature groups...")
    rows: list[dict[str, float | str]] = [
        {
            "group": "None (full model)",
            "removed_features": "---",
            "recall": float(base_recall),
            "delta_vs_full_model": float("nan"),
        }
    ]

    for idx, (group_name, removed_cols) in enumerate(FEATURE_GROUPS.items(), start=1):
        logger.info(f"  [{idx}/{len(FEATURE_GROUPS)}] Ablating {group_name}...")
        keep_cols = [col for col in FEATURE_COLS if col not in removed_cols]
        model = build_model()
        model.fit(x_train[keep_cols], y_train)
        y_pred = np.asarray(model.predict(x_test[keep_cols])).astype(int).ravel()
        recall = recall_score(y_test, y_pred, zero_division=0)
        delta = recall - base_recall
        logger.info(f"    Recall: {recall:.2%} (delta: {delta:+.2%})")

        rows.append(
            {
                "group": group_name,
                "removed_features": ", ".join(removed_cols),
                "recall": float(recall),
                "delta_vs_full_model": float(delta),
            }
        )

    result_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    logger.info(f"\n✓ Ablation study complete. Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
