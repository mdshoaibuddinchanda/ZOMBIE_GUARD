"""
variant_recall.py
ZombieGuard - Per-variant recall breakdown.

For each of the 8 synthetic attack variants (A-H) plus real-world samples,
evaluates the trained XGBoost model and reports:
  Variant | Name | N (test) | TP | FN | Recall | Primary driving feature

This table demonstrates that the taxonomy has diagnostic power —
each variant is caught by a different combination of features.

Outputs:
  paper/figures/table7_variant_recall.csv
  paper/figures/table7_variant_recall.png
  paper/figures/table7_variant_recall.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import extract_features
from src.classifier import FEATURE_COLS, predict

# -- Paths -------------------------------------------------------------------
MALICIOUS_DIR = "data/raw/malicious"
MODEL_PATH    = "models/lgbm_model.pkl"
CSV_DIR       = "paper/figures/csv"
PNG_DIR       = "paper/figures/png"
PDF_DIR       = "paper/figures/pdf"
OUTPUT_CSV    = f"{CSV_DIR}/table7_variant_recall.csv"
OUTPUT_STEM   = "table7_variant_recall"

# -- Variant definitions -----------------------------------------------------
# Maps file prefix -> (variant_id, display_name, primary_feature)
VARIANTS = [
    ("zombie_A_classic",          "A", "Classic Zombie ZIP",         "method_mismatch + declared_vs_entropy_flag"),
    ("zombie_B_method_only",      "B", "Method-only mismatch",       "method_mismatch"),
    ("zombie_C_gootloader",       "C", "Gootloader concatenation",   "eocd_count > 1"),
    ("zombie_D_multifile",        "D", "Multi-file decoy",           "suspicious_entry_ratio"),
    ("zombie_E_crc_mismatch",     "E", "CRC32 mismatch",             "any_crc_mismatch"),
    ("zombie_F_extra_noise",      "F", "Extra field noise",          "declared_vs_entropy_flag"),
    ("zombie_G_high_compression", "G", "High compression gap",       "declared_vs_entropy_flag + entropy"),
    ("zombie_H_size_mismatch",    "H", "Size field mismatch",        "method_mismatch + entropy"),
]

# Variant I only appears in real-world data (single sample, undefined method)
VARIANT_I = ("I", "Undefined method code", "lf_unknown_method")

# -- Color palette -----------------------------------------------------------
PRIMARY_BLUE  = "#0D4EA6"
SUCCESS_GREEN = "#2D6A4F"
AMBER         = "#D4820A"
PRIMARY_RED   = "#B22222"
LIGHT_GRAY    = "#F5F5F5"
MED_GRAY      = "#CCCCCC"
DARK_GRAY     = "#444444"
LIGHT_RED_BG  = "#FFE8E8"
LIGHT_GREEN_BG = "#E8F5E9"


# -- Feature extraction ------------------------------------------------------
def extract_variant_files(malicious_dir: str) -> dict[str, list[str]]:
    """Group malicious ZIP paths by variant prefix."""
    groups: dict[str, list[str]] = {prefix: [] for prefix, *_ in VARIANTS}

    for fname in sorted(os.listdir(malicious_dir)):
        if not fname.endswith(".zip"):
            continue
        for prefix, *_ in VARIANTS:
            if fname.startswith(prefix):
                groups[prefix].append(os.path.join(malicious_dir, fname))
                break

    return groups


def evaluate_variant(
    model,
    file_paths: list[str],
    threshold: float = 0.5,
) -> tuple[int, int, int, list[dict]]:
    """
    Run model on all files for one variant.
    Returns (total, TP, FN, feature_rows).
    """
    tp = fn = 0
    feature_rows = []

    for fpath in file_paths:
        features = extract_features(fpath)
        result = predict(model, features)
        features["_prob"] = result["probability"]
        features["_pred"] = result["label"]
        feature_rows.append(features)

        if result["label"] == 1:
            tp += 1
        else:
            fn += 1

    return len(file_paths), tp, fn, feature_rows


def dominant_feature_stats(feature_rows: list[dict], fn_rows: list[dict]) -> dict:
    """
    Compute mean values of key discriminating features across all variant samples.
    Used to verify which features are actually firing.
    """
    if not feature_rows:
        return {}

    keys = [
        "method_mismatch", "declared_vs_entropy_flag", "eocd_count",
        "suspicious_entry_ratio", "any_crc_mismatch", "lf_unknown_method",
        "data_entropy_shannon",
    ]
    means = {}
    for k in keys:
        vals = [float(r.get(k, 0)) for r in feature_rows]
        means[k] = round(np.mean(vals), 3)

    # Mean probability score
    probs = [r.get("_prob", 0.0) for r in feature_rows]
    means["mean_prob"] = round(np.mean(probs), 4)

    # FN mean prob (how close were the misses?)
    if fn_rows:
        fn_probs = [r.get("_prob", 0.0) for r in fn_rows]
        means["fn_mean_prob"] = round(np.mean(fn_probs), 4)
    else:
        means["fn_mean_prob"] = None

    return means


# -- Main evaluation ---------------------------------------------------------
def run_variant_breakdown(model_path: str = MODEL_PATH) -> pd.DataFrame:
    """Evaluate XGBoost on every variant and build the breakdown table."""
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"Scanning: {MALICIOUS_DIR}\n")
    groups = extract_variant_files(MALICIOUS_DIR)

    rows = []

    for prefix, vid, name, primary_feature in VARIANTS:
        files = groups[prefix]
        if not files:
            print(f"  WARNING: No files found for variant {vid} (prefix={prefix})")
            rows.append({
                "variant": vid,
                "name": name,
                "n_total": 0,
                "n_test": 0,
                "TP": 0,
                "FN": 0,
                "recall": None,
                "mean_prob": None,
                "fn_mean_prob": None,
                "primary_feature": primary_feature,
            })
            continue

        total, tp, fn, feat_rows = evaluate_variant(model, files)
        fn_rows = [r for r in feat_rows if r["_pred"] == 0]
        stats = dominant_feature_stats(feat_rows, fn_rows)
        recall = round(tp / total, 4) if total > 0 else 0.0

        print(
            f"  Variant {vid} ({name:<28})  "
            f"N={total:>4}  TP={tp:>4}  FN={fn:>3}  "
            f"Recall={recall:.4f}  mean_p={stats.get('mean_prob', 0):.4f}"
        )

        rows.append({
            "variant": vid,
            "name": name,
            "n_total": total,
            "n_test": total,
            "TP": tp,
            "FN": fn,
            "recall": recall,
            "mean_prob": stats.get("mean_prob"),
            "fn_mean_prob": stats.get("fn_mean_prob"),
            "primary_feature": primary_feature,
        })

    # Variant I — real-world MalwareBazaar samples with undefined method code (99)
    _append_variant_i(model, rows)

    df = pd.DataFrame(rows)
    return df


def _append_variant_i(model, rows: list) -> None:
    """
    Variant I (undefined method code) appears only in real-world data.
    Check features.csv for lf_unknown_method=1 samples.
    """
    feat_path = "data/processed/features.csv"
    label_path = "data/processed/labels.csv"

    if not os.path.exists(feat_path) or not os.path.exists(label_path):
        rows.append({
            "variant": "I",
            "name": "Undefined method code",
            "n_total": 0, "n_test": 0, "TP": 0, "FN": 0,
            "recall": None, "mean_prob": None, "fn_mean_prob": None,
            "primary_feature": "lf_unknown_method",
        })
        return

    feat_df = pd.read_csv(feat_path)
    label_df = pd.read_csv(label_path)
    merged = feat_df.merge(label_df, on="filename")

    # Variant I: malicious samples with unknown method code
    vi_df = merged[
        (merged["label"] == 1) &
        (merged["lf_unknown_method"].astype(float) > 0)
    ]

    if vi_df.empty:
        print(f"  Variant I (Undefined method)          N=   0  (no samples in dataset)")
        rows.append({
            "variant": "I",
            "name": "Undefined method code",
            "n_total": 0, "n_test": 0, "TP": 0, "FN": 0,
            "recall": None, "mean_prob": None, "fn_mean_prob": None,
            "primary_feature": "lf_unknown_method",
        })
        return

    available = [c for c in FEATURE_COLS if c in vi_df.columns]
    bool_cols = ["method_mismatch", "declared_vs_entropy_flag", "any_crc_mismatch", "is_encrypted"]
    for col in bool_cols:
        if col in vi_df.columns:
            vi_df = vi_df.copy()
            vi_df[col] = vi_df[col].astype(int)

    X = vi_df[available].fillna(0).astype(float)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    tp = int(preds.sum())
    fn = int(len(preds) - tp)
    recall = round(tp / len(preds), 4) if len(preds) > 0 else 0.0
    mean_prob = round(float(np.mean(probs)), 4)

    print(
        f"  Variant I (Undefined method          )  "
        f"N={len(preds):>4}  TP={tp:>4}  FN={fn:>3}  "
        f"Recall={recall:.4f}  mean_p={mean_prob:.4f}"
    )

    rows.append({
        "variant": "I",
        "name": "Undefined method code",
        "n_total": len(preds),
        "n_test": len(preds),
        "TP": tp,
        "FN": fn,
        "recall": recall,
        "mean_prob": mean_prob,
        "fn_mean_prob": None,
        "primary_feature": "lf_unknown_method",
    })


# -- Figure generation -------------------------------------------------------
def generate_table_figure(df: pd.DataFrame, png_dir: str, pdf_dir: str) -> tuple[str, str]:
    """
    Tables are saved as CSV only (no image output per project policy).
    This function is kept for API compatibility but does not produce image files.
    """
    print(f"  (Table '{OUTPUT_STEM}' saved as CSV only — no image output for tables)")
    return "", ""


def generate_recall_bar(df: pd.DataFrame, png_dir: str, pdf_dir: str) -> str:
    """Horizontal bar chart of recall per variant — easy to read at a glance."""
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    plot_df = df[df["recall"].notna() & (df["n_test"] > 0)].copy()
    labels = [f"{r['variant']} — {r['name']}" for _, r in plot_df.iterrows()]
    recalls = plot_df["recall"].tolist()

    colors = []
    for r in recalls:
        if r >= 1.0:
            colors.append(SUCCESS_GREEN)
        elif r >= 0.90:
            colors.append(AMBER)
        else:
            colors.append(PRIMARY_RED)

    fig, ax = plt.subplots(figsize=(7.0, max(3.5, len(labels) * 0.42)), constrained_layout=False)
    y = np.arange(len(labels))
    bars = ax.barh(y, recalls, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_title(
        "Figure 6 — Per-Variant Recall (XGBoost)",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(x=1.0, linestyle="--", linewidth=0.9, color=MED_GRAY)
    ax.axvline(x=0.90, linestyle=":", linewidth=0.8, color=AMBER, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4, color=MED_GRAY)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=4, color=DARK_GRAY)

    fig.subplots_adjust(left=0.32, right=0.97, top=0.92, bottom=0.10)

    out_path = str(Path(png_dir) / "fig6_variant_recall_chart.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(str(Path(pdf_dir) / "fig6_variant_recall_chart.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print("PER-VARIANT RECALL BREAKDOWN")
    print(f"{'='*80}")
    print(f"{'ID':<4} {'Name':<30} {'N':>5} {'TP':>5} {'FN':>5} {'Recall':>8}  Primary Feature")
    print("-" * 80)
    for _, row in df.iterrows():
        recall_str = f"{row['recall']:.4f}" if row["recall"] is not None else "  N/A"
        n = int(row["n_test"]) if row["n_test"] else 0
        print(
            f"{row['variant']:<4} {row['name']:<30} {n:>5} "
            f"{int(row['TP']):>5} {int(row['FN']):>5} {recall_str:>8}  "
            f"{row['primary_feature']}"
        )
    print("=" * 80)

    # Overall weighted recall
    valid = df[df["recall"].notna() & (df["n_test"] > 0)]
    if not valid.empty:
        total_n = valid["n_test"].sum()
        total_tp = valid["TP"].sum()
        overall = total_tp / total_n if total_n > 0 else 0
        print(f"\nOverall (weighted): {int(total_tp)}/{int(total_n)} = {overall:.4f} recall")


if __name__ == "__main__":
    print("ZombieGuard — Per-Variant Recall Breakdown")
    print("=" * 55)

    df = run_variant_breakdown()

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV: {OUTPUT_CSV}")

    print_summary(df)

    generate_table_figure(df, PNG_DIR, PDF_DIR)
    generate_recall_bar(df, PNG_DIR, PDF_DIR)

    print("\nDone. Outputs:")
    print(f"  CSV  → {OUTPUT_CSV}")
    print(f"  PNG  → {PNG_DIR}/fig6_variant_recall_chart.png")
    print(f"  PDF  → {PDF_DIR}/fig6_variant_recall_chart.pdf")
