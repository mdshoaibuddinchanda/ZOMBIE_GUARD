"""Run feature-group ablation on the paper holdout split and save CSV results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import FEATURE_COLS


FEATURE_GROUPS: dict[str, list[str]] = {
    "Entropy features (4, 5, 6)": [
        "data_entropy_shannon",
        "data_entropy_renyi",
        "declared_vs_entropy_flag",
    ],
    "Suspicious entry group (3, 9, 10)": [
        "method_mismatch",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
    ],
    "Header method fields (1, 2)": [
        "lf_compression_method",
        "cd_compression_method",
    ],
    "CRC mismatch (11)": ["any_crc_mismatch"],
    "EOCD count (7)": ["eocd_count"],
    "Encryption flag (12)": ["is_encrypted"],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for output location and input artifacts."""
    parser = argparse.ArgumentParser(
        description="Run ZombieGuard feature-group ablation and save table CSV."
    )
    parser.add_argument(
        "--features",
        default="data/processed/features.csv",
        help="Path to features CSV.",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/labels.csv",
        help="Path to labels CSV.",
    )
    parser.add_argument(
        "--model",
        default="models/xgboost_model.pkl",
        help="Path to trained XGBoost model.",
    )
    parser.add_argument(
        "--output",
        default="paper/figures/table5_feature_ablation.csv",
        help="CSV output path for ablation table.",
    )
    return parser.parse_args()


def load_holdout_split(
    features_path: str,
    labels_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce the exact paper split using merged filename-aligned records."""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    merged = features_df.merge(labels_df, on="filename", how="inner")
    for col in [
        "method_mismatch",
        "declared_vs_entropy_flag",
        "lf_crc_valid",
        "any_crc_mismatch",
        "is_encrypted",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)

    x = merged[FEATURE_COLS].copy()
    y = merged["label"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def build_model() -> XGBClassifier:
    """Build the model used for each ablation retraining run."""
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


def run_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_path: str,
) -> pd.DataFrame:
    """Compute full-model recall and recall deltas per removed feature group."""
    full_model = joblib.load(model_path)
    base_recall = recall_score(y_test, full_model.predict(x_test))

    rows: list[dict[str, str | float]] = [
        {
            "group": "None (full model)",
            "removed_features": "---",
            "recall": float(base_recall),
            "delta_vs_full_model": float("nan"),
        }
    ]

    for group_name, cols in FEATURE_GROUPS.items():
        remaining = [feature for feature in FEATURE_COLS if feature not in cols]
        model = build_model()
        model.fit(x_train[remaining], y_train)
        recall = recall_score(y_test, model.predict(x_test[remaining]))
        rows.append(
            {
                "group": group_name,
                "removed_features": ", ".join(cols),
                "recall": float(recall),
                "delta_vs_full_model": float(recall - base_recall),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Run ablation and persist CSV for paper table reproducibility."""
    args = parse_args()
    x_train, x_test, y_train, y_test = load_holdout_split(args.features, args.labels)
    results = run_ablation(x_train, x_test, y_train, y_test, args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(f"Test set size: {len(x_test)} samples")
    print(f"Malicious in test: {int(y_test.sum())}")
    print(f"Saved ablation CSV: {out_path}")
    print()

    for _, row in results.iterrows():
        if row["group"] == "None (full model)":
            print(f"{row['group']} | {row['recall']:.4f} | ---")
            continue
        print(
            f"{row['group']} | {row['recall']:.4f} | "
            f"{row['delta_vs_full_model']:+.4f}"
        )


if __name__ == "__main__":
    main()
