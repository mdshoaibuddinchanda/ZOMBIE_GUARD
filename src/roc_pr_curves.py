"""
roc_pr_curves.py
EXPERIMENT 4 — ROC Curve and Precision-Recall Curve

Plots ROC and PR curves for ZombieGuard XGBoost vs rule-based baseline
on the same axes. Both curves use the same 80/20 synthetic holdout split.

Outputs:
  paper/figures/csv/table_roc_pr_auc.csv
  paper/figures/png/fig8_roc_curve.png
  paper/figures/pdf/fig8_roc_curve.pdf
  paper/figures/png/fig9_pr_curve.png
  paper/figures/pdf/fig9_pr_curve.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import FEATURE_COLS

FEATURES_PATH = "data/processed/features.csv"
LABELS_PATH   = "data/processed/labels.csv"
MODEL_PATH    = "models/lgbm_model.pkl"CSV_DIR       = "paper/figures/csv"
PNG_DIR       = "paper/figures/png"
PDF_DIR       = "paper/figures/pdf"
RANDOM_STATE  = 42

PRIMARY_BLUE   = "#0D4EA6"
PRIMARY_RED    = "#B22222"
SUCCESS_GREEN  = "#2D6A4F"
AMBER          = "#D4820A"
MED_GRAY       = "#CCCCCC"
DARK_GRAY      = "#444444"


def configure_style():
    available = {f.name for f in fm.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": font,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_data():
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df   = pd.read_csv(LABELS_PATH)
    merged = features_df.merge(labels_df, on="filename")
    bool_cols = ["method_mismatch", "declared_vs_entropy_flag",
                 "lf_crc_valid", "any_crc_mismatch", "is_encrypted"]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)
    available = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available].fillna(0).astype(float)
    y = merged["label"].astype(int)
    return X, y


def rule_based_score(X: pd.DataFrame) -> np.ndarray:
    """
    Replicate the rule-based baseline scoring.
    Score = sum of triggered signals (0-4), normalised to [0,1].
    Signals: method_mismatch, declared_vs_entropy_flag, eocd_count>1, any_crc_mismatch
    """
    score = np.zeros(len(X))
    if "method_mismatch" in X.columns:
        score += X["method_mismatch"].astype(float).values
    if "declared_vs_entropy_flag" in X.columns:
        score += X["declared_vs_entropy_flag"].astype(float).values
    if "eocd_count" in X.columns:
        score += (X["eocd_count"] > 1).astype(float).values
    if "any_crc_mismatch" in X.columns:
        score += X["any_crc_mismatch"].astype(float).values
    return score / 4.0


def _save(fig, stem):
    Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    png = f"{PNG_DIR}/{stem}.png"
    pdf = f"{PDF_DIR}/{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")


def generate_roc_curve(fpr_xgb, tpr_xgb, auc_xgb,
                       fpr_base, tpr_base, auc_base):
    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)

    ax.plot(fpr_xgb, tpr_xgb, color=PRIMARY_BLUE, lw=2,
            label=f"ZombieGuard LightGBM (AUC = {auc_xgb:.4f})")
    ax.plot(fpr_base, tpr_base, color=AMBER, lw=2, linestyle="--",
            label=f"Rule-based baseline (AUC = {auc_base:.4f})")
    ax.plot([0, 1], [0, 1], color=MED_GRAY, lw=1, linestyle=":", label="Random classifier")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("Figure 8 — ROC Curve: ZombieGuard vs Rule-Based Baseline", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.4, color=MED_GRAY)

    _save(fig, "fig8_roc_curve")


def generate_pr_curve(prec_xgb, rec_xgb, ap_xgb,
                      prec_base, rec_base, ap_base,
                      baseline_prevalence):
    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)

    ax.plot(rec_xgb, prec_xgb, color=PRIMARY_BLUE, lw=2,
            label=f"ZombieGuard LightGBM (AP = {ap_xgb:.4f})")
    ax.plot(rec_base, prec_base, color=AMBER, lw=2, linestyle="--",
            label=f"Rule-based baseline (AP = {ap_base:.4f})")
    ax.axhline(y=baseline_prevalence, color=MED_GRAY, lw=1, linestyle=":",
               label=f"No-skill baseline ({baseline_prevalence:.3f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Figure 9 — Precision-Recall Curve: ZombieGuard vs Rule-Based Baseline", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.4, color=MED_GRAY)

    _save(fig, "fig9_pr_curve")


def main():
    print("EXPERIMENT 4 — ROC and Precision-Recall Curves")
    print("=" * 55)

    configure_style()

    X, y = load_data()
    print(f"Dataset: {len(X)} samples  malicious={y.sum()}  benign={(y==0).sum()}")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    xgb_proba  = model.predict_proba(X_test)[:, 1]
    base_score = rule_based_score(X_test)

    # ROC
    fpr_xgb,  tpr_xgb,  _ = roc_curve(y_test, xgb_proba)
    fpr_base, tpr_base, _ = roc_curve(y_test, base_score)
    auc_xgb  = roc_auc_score(y_test, xgb_proba)
    auc_base = roc_auc_score(y_test, base_score)

    # PR
    prec_xgb,  rec_xgb,  _ = precision_recall_curve(y_test, xgb_proba)
    prec_base, rec_base, _ = precision_recall_curve(y_test, base_score)
    ap_xgb  = average_precision_score(y_test, xgb_proba)
    ap_base = average_precision_score(y_test, base_score)
    prevalence = float(y_test.sum()) / len(y_test)

    print(f"\nLightGBM — ROC-AUC: {auc_xgb:.4f}  AP: {ap_xgb:.4f}")
    print(f"Baseline — ROC-AUC: {auc_base:.4f}  AP: {ap_base:.4f}")
    print(f"Class prevalence in test set: {prevalence:.4f}")

    # Save summary CSV
    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"model": "ZombieGuard XGBoost", "roc_auc": round(auc_xgb, 4), "avg_precision": round(ap_xgb, 4)},
        {"model": "Rule-based baseline", "roc_auc": round(auc_base, 4), "avg_precision": round(ap_base, 4)},
    ])
    csv_path = f"{CSV_DIR}/table_roc_pr_auc.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    generate_roc_curve(fpr_xgb, tpr_xgb, auc_xgb, fpr_base, tpr_base, auc_base)
    generate_pr_curve(prec_xgb, rec_xgb, ap_xgb, prec_base, rec_base, ap_base, prevalence)

    print("\nDone.")


if __name__ == "__main__":
    main()
