"""
entropy_distribution.py
EXPERIMENT 5 — Entropy Distribution Plot

Plots overlapping Shannon entropy histograms for malicious vs benign samples,
with a vertical line at the 7.0 threshold used by declared_vs_entropy_flag.
Proves the threshold is empirically grounded, not arbitrary.

Outputs:
  paper/figures/csv/table_entropy_stats.csv
  paper/figures/png/fig10_entropy_distribution.png
  paper/figures/pdf/fig10_entropy_distribution.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import FEATURE_COLS

FEATURES_PATH = "data/processed/features.csv"
LABELS_PATH   = "data/processed/labels.csv"
CSV_DIR       = "paper/figures/csv"
PNG_DIR       = "paper/figures/png"
PDF_DIR       = "paper/figures/pdf"

PRIMARY_BLUE  = "#0D4EA6"
PRIMARY_RED   = "#B22222"
AMBER         = "#D4820A"
MED_GRAY      = "#CCCCCC"
DARK_GRAY     = "#444444"
THRESHOLD     = 7.0


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


def main():
    print("EXPERIMENT 5 — Entropy Distribution Plot")
    print("=" * 55)

    configure_style()

    features_df = pd.read_csv(FEATURES_PATH)
    labels_df   = pd.read_csv(LABELS_PATH)
    merged = features_df.merge(labels_df, on="filename")

    mal_entropy = merged[merged["label"] == 1]["data_entropy_shannon"].dropna().values
    ben_entropy = merged[merged["label"] == 0]["data_entropy_shannon"].dropna().values

    print(f"Malicious samples: {len(mal_entropy)}")
    print(f"  mean={mal_entropy.mean():.4f}  std={mal_entropy.std():.4f}")
    print(f"  min={mal_entropy.min():.4f}  max={mal_entropy.max():.4f}")
    print(f"  % above {THRESHOLD}: {100*(mal_entropy >= THRESHOLD).mean():.1f}%")
    print(f"\nBenign samples: {len(ben_entropy)}")
    print(f"  mean={ben_entropy.mean():.4f}  std={ben_entropy.std():.4f}")
    print(f"  min={ben_entropy.min():.4f}  max={ben_entropy.max():.4f}")
    print(f"  % above {THRESHOLD}: {100*(ben_entropy >= THRESHOLD).mean():.1f}%")
    print(f"\nNote: {100*(ben_entropy >= THRESHOLD).mean():.1f}% of benign samples also exceed the")
    print(f"  threshold — entropy alone is insufficient for detection. The 7.0 threshold")
    print(f"  is one signal among 12 features; the ML model resolves the overlap region.")

    # Save stats CSV
    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame([
        {
            "class": "Malicious",
            "n": len(mal_entropy),
            "mean": round(mal_entropy.mean(), 4),
            "std": round(mal_entropy.std(), 4),
            "min": round(mal_entropy.min(), 4),
            "max": round(mal_entropy.max(), 4),
            "pct_above_threshold": round(100 * (mal_entropy >= THRESHOLD).mean(), 2),
        },
        {
            "class": "Benign",
            "n": len(ben_entropy),
            "mean": round(ben_entropy.mean(), 4),
            "std": round(ben_entropy.std(), 4),
            "min": round(ben_entropy.min(), 4),
            "max": round(ben_entropy.max(), 4),
            "pct_above_threshold": round(100 * (ben_entropy >= THRESHOLD).mean(), 2),
        },
    ])
    csv_path = f"{CSV_DIR}/table_entropy_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot
    bins = np.linspace(0, 8, 65)  # 0.125-bit bins

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)

    ax.hist(ben_entropy, bins=bins, alpha=0.55, color=PRIMARY_BLUE,
            label=f"Benign (n={len(ben_entropy)})", density=True, edgecolor="none")
    ax.hist(mal_entropy, bins=bins, alpha=0.55, color=PRIMARY_RED,
            label=f"Malicious (n={len(mal_entropy)})", density=True, edgecolor="none")

    ax.axvline(x=THRESHOLD, color=AMBER, lw=2, linestyle="--",
               label=f"Threshold = {THRESHOLD} bits/byte")

    # Annotate threshold
    ymax = ax.get_ylim()[1]
    ax.text(THRESHOLD + 0.05, ymax * 0.88,
            f"declared_vs_entropy_flag\nthreshold = {THRESHOLD}",
            fontsize=7.5, color=AMBER, va="top")

    ax.set_xlabel("Shannon Entropy (bits/byte)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Figure 10 — Shannon Entropy Distribution: Malicious vs Benign", fontsize=10)
    ax.set_xlim(0, 8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4, color=MED_GRAY)

    _save(fig, "fig10_entropy_distribution")
    print("\nDone.")


if __name__ == "__main__":
    main()
