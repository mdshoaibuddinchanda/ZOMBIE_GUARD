"""
multi_baseline.py
ZombieGuard - Multiple ML baseline comparison.

Runs 5 classifiers on the same 12 features and identical 80/20 holdout split:
  1. Logistic Regression
  2. Linear SVM
  3. Random Forest
  4. LightGBM
  5. XGBoost (existing champion)

Produces:
  paper/figures/table6_multi_baseline_comparison.csv
  paper/figures/table6_multi_baseline_comparison.png
  paper/figures/table6_multi_baseline_comparison.pdf

Directly addresses "insufficient comparative analysis" by proving XGBoost
superiority rather than assuming it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import FEATURE_COLS

# -- Paths -------------------------------------------------------------------
FEATURES_PATH = "data/processed/features.csv"
LABELS_PATH   = "data/processed/labels.csv"
CSV_DIR       = "paper/figures/csv"
PNG_DIR       = "paper/figures/png"
PDF_DIR       = "paper/figures/pdf"
OUTPUT_CSV    = f"{CSV_DIR}/table6_multi_baseline_comparison.csv"
OUTPUT_STEM   = "table6_multi_baseline_comparison"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_SPLITS     = 5

# -- Color palette (matches paper style) ------------------------------------
PRIMARY_BLUE = "#0D4EA6"
SECONDARY_BLUE = "#4A90D9"
SUCCESS_GREEN = "#2D6A4F"
AMBER = "#D4820A"
PRIMARY_RED = "#B22222"
LIGHT_GRAY = "#F5F5F5"
MED_GRAY = "#CCCCCC"
DARK_GRAY = "#444444"


# -- Data loading ------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge features/labels, cast booleans to int."""
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    merged = features_df.merge(labels_df, on="filename", how="inner")

    bool_cols = [
        "method_mismatch",
        "declared_vs_entropy_flag",
        "lf_crc_valid",
        "any_crc_mismatch",
        "is_encrypted",
    ]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)

    available = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available].fillna(0).astype(float)
    y = merged["label"].astype(int)
    return X, y


# -- Model definitions -------------------------------------------------------
def build_models() -> dict:
    """
    Return all five classifiers.
    LR and SVM are wrapped in a StandardScaler pipeline since they are
    sensitive to feature scale. RF and XGBoost are scale-invariant.
    LinearSVC does not natively output probabilities, so it is wrapped
    in CalibratedClassifierCV for AUC computation.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                C=1.0,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    C=1.0,
                ),
                cv=3,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "LightGBM": _build_lgbm(),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _build_lgbm():
    """Build LightGBM classifier with graceful fallback if not installed."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("WARNING: LightGBM not installed. Substituting extra Random Forest.")
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=RANDOM_STATE + 1,
            n_jobs=-1,
            class_weight="balanced",
        )


# -- Metrics -----------------------------------------------------------------
def compute_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "TP": int(cm[1, 1]),
        "FP": int(cm[0, 1]),
        "TN": int(cm[0, 0]),
        "FN": int(cm[1, 0]),
    }


# -- Cross-validation --------------------------------------------------------
def cross_validate_model(name: str, model, X_train, y_train) -> dict:
    """Run 5-fold stratified CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_recalls, fold_f1s, fold_aucs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Re-build a fresh model for each fold to avoid state leakage
        fold_model = build_models()[name]
        fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val)[:, 1]

        fold_recalls.append(recall_score(y_val, y_pred, zero_division=0))
        fold_f1s.append(f1_score(y_val, y_pred, zero_division=0))
        fold_aucs.append(roc_auc_score(y_val, y_prob))

    return {
        "cv_recall_mean": round(float(np.mean(fold_recalls)), 4),
        "cv_recall_std": round(float(np.std(fold_recalls)), 4),
        "cv_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "cv_auc_mean": round(float(np.mean(fold_aucs)), 4),
    }


# -- Hard test set loader ----------------------------------------------------
def load_hard_test() -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """
    Load the hard test set from data/hard_test/ if it exists.
    This set was built to remove the easy EOCD signal (ratio 1.18x),
    forcing models to use entropy, method mismatch, CRC, and structural
    features together. It is the most discriminating evaluation.
    Returns (None, None) if the hard test set has not been built yet.
    """
    hard_ev_dir = Path("data/hard_test/evasion")
    hard_nev_dir = Path("data/hard_test/non_evasion")

    if not hard_ev_dir.exists() or not hard_nev_dir.exists():
        return None, None

    try:
        from src.extractor import extract_features
    except ImportError:
        return None, None

    rows = []
    for fpath in sorted(hard_ev_dir.glob("*.zip")):
        f = extract_features(str(fpath))
        f["label"] = 1
        rows.append(f)
    for fpath in sorted(hard_nev_dir.glob("*.zip")):
        f = extract_features(str(fpath))
        f["label"] = 0
        rows.append(f)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).astype(float)
    y = df["label"].astype(int)
    print(f"Hard test set: {len(X)} samples  |  evasion={int(y.sum())}  non-evasion={int((y==0).sum())}")
    return X, y


# -- Main evaluation ---------------------------------------------------------
def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Train all models on the synthetic 80/20 split.
    Evaluate on both:
      1. Standard holdout (same split as classifier.py)
      2. Hard test set (EOCD-resistant, multi-feature required) if available
    """
    print("Loading data...")
    X, y = load_data()

    print(f"Dataset: {len(X)} samples  |  malicious={int(y.sum())}  benign={int((y==0).sum())}")
    print(f"Features: {list(X.columns)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # Load hard test set once
    X_hard, y_hard = load_hard_test()

    models = build_models()
    results_standard = []
    results_hard = []
    trained_models = {}

    for name, model in models.items():
        print(f"{'='*55}")
        print(f"Training: {name}")

        # Cross-validation on training split
        print(f"  Running {N_SPLITS}-fold CV...")
        cv_stats = cross_validate_model(name, model, X_train, y_train)
        print(
            f"  CV recall={cv_stats['cv_recall_mean']:.4f}±{cv_stats['cv_recall_std']:.4f}  "
            f"f1={cv_stats['cv_f1_mean']:.4f}  auc={cv_stats['cv_auc_mean']:.4f}"
        )

        # Final fit on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # -- Standard holdout evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(name, y_test, y_pred, y_prob)
        metrics.update(cv_stats)
        results_standard.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Standard holdout:")
        print(f"    Acc={metrics['accuracy']}  Prec={metrics['precision']}  "
              f"Rec={metrics['recall']}  F1={metrics['f1']}  AUC={metrics['roc_auc']}")
        print(f"    TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        # -- Hard test evaluation (if available)
        if X_hard is not None:
            # Align columns in case hard test has different available features
            X_hard_aligned = X_hard.reindex(columns=X_train.columns, fill_value=0)
            y_pred_h = model.predict(X_hard_aligned)
            y_prob_h = model.predict_proba(X_hard_aligned)[:, 1]
            hard_metrics = compute_metrics(name, y_hard, y_pred_h, y_prob_h)
            results_hard.append(hard_metrics)
            cm_h = confusion_matrix(y_hard, y_pred_h)
            print(f"\n  Hard test set (EOCD-resistant):")
            print(f"    Acc={hard_metrics['accuracy']}  Prec={hard_metrics['precision']}  "
                  f"Rec={hard_metrics['recall']}  F1={hard_metrics['f1']}  AUC={hard_metrics['roc_auc']}")
            print(f"    TN={cm_h[0,0]}  FP={cm_h[0,1]}  FN={cm_h[1,0]}  TP={cm_h[1,1]}")

        print()

    df_standard = pd.DataFrame(results_standard)
    df_hard = pd.DataFrame(results_hard) if results_hard else None
    return df_standard, df_hard


# -- Figure generation -------------------------------------------------------
def generate_table_figure(df: pd.DataFrame, png_dir: str, pdf_dir: str, stem: str, title: str) -> tuple[str, str]:
    """
    Tables are saved as CSV only (no image output per project policy).
    This function is kept for API compatibility but does not produce image files.
    """
    print(f"  (Table '{stem}' saved as CSV only — no image output for tables)")
    return "", ""



def generate_bar_chart(df: pd.DataFrame, png_dir: str, pdf_dir: str, stem: str, title: str) -> str:
    """Grouped bar chart comparing recall and AUC across all 5 models."""
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    models = df["model"].tolist()
    recall_vals = df["recall"].tolist()
    auc_vals = df["roc_auc"].tolist()

    x = np.arange(len(models))

    bar_colors_recall = [SUCCESS_GREEN if m == "LightGBM" else SECONDARY_BLUE for m in models]
    bar_colors_auc = [SUCCESS_GREEN if m == "LightGBM" else AMBER for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=False)

    for ax, vals, colors, subplot_title, ylabel in [
        (axes[0], recall_vals, bar_colors_recall, "Recall (Malicious Detection Rate)", "Recall"),
        (axes[1], auc_vals, bar_colors_auc, "ROC-AUC", "ROC-AUC"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(subplot_title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4, color=MED_GRAY)
        ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=3, color=DARK_GRAY)
        ax.axhline(y=0.99, linestyle="--", linewidth=0.8, color=PRIMARY_RED, alpha=0.5)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.22, wspace=0.28)

    out_path = str(Path(png_dir) / f"{stem}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(str(Path(pdf_dir) / f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def print_summary_table(df: pd.DataFrame, label: str) -> None:
    """Print a clean console summary table."""
    print(f"\n{'='*75}")
    print(f"MULTI-MODEL COMPARISON — {label}")
    print(f"{'='*75}")
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'FP':>5} {'FN':>5}")
    print("-" * 75)
    for _, row in df.iterrows():
        marker = " <-- LightGBM (primary)" if row["model"] == "LightGBM" else ""
        print(
            f"{row['model']:<22} "
            f"{row['accuracy']:>7.4f} "
            f"{row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} "
            f"{row['f1']:>7.4f} "
            f"{row['roc_auc']:>7.4f} "
            f"{int(row['FP']):>5} "
            f"{int(row['FN']):>5}"
            f"{marker}"
        )
    print("=" * 75)


if __name__ == "__main__":
    print("ZombieGuard — Multi-Model Baseline Comparison")
    print("=" * 55)

    df_standard, df_hard = run_comparison()

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    # -- Standard holdout outputs
    df_standard.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print_summary_table(df_standard, "Standard holdout (80/20 synthetic split)")

    generate_table_figure(df_standard, PNG_DIR, PDF_DIR, stem=OUTPUT_STEM,
        title="Table 6 — Multi-Model Comparison (same 12 features, identical 80/20 holdout split)")
    generate_bar_chart(df_standard, PNG_DIR, PDF_DIR, stem="fig5_multi_baseline_chart",
        title="ZombieGuard — Multi-Model Comparison: Recall and ROC-AUC")

    # -- Hard test outputs (if available)
    if df_hard is not None:
        hard_csv = f"{CSV_DIR}/table6b_multi_baseline_hard_test.csv"
        df_hard.to_csv(hard_csv, index=False)
        print(f"\nSaved CSV: {hard_csv}")
        print_summary_table(df_hard, "Hard test set (EOCD-resistant, multi-feature required)")

        generate_table_figure(df_hard, PNG_DIR, PDF_DIR, stem="table6b_multi_baseline_hard_test",
            title="Table 6b — Multi-Model Comparison on Hard Test Set (EOCD ratio 1.18x)")
        generate_bar_chart(df_hard, PNG_DIR, PDF_DIR, stem="fig5b_multi_baseline_hard_chart",
            title="ZombieGuard — Hard Test Set: Recall and ROC-AUC (multi-feature required)")
    else:
        print("\nNote: Hard test set not found. Run data/scripts/build_hard_testset.py to generate it.")

    print("\nDone. Outputs:")
    print(f"  CSV  → {CSV_DIR}/")
    print(f"  PNG  → {PNG_DIR}/")
    print(f"  PDF  → {PDF_DIR}/")
