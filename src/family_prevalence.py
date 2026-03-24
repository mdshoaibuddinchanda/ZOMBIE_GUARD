"""
family_prevalence.py
EXPERIMENT 6 — Per-Family Prevalence Breakdown

Joins realworld_labels.csv with bazaar_timestamps.csv to get malware family
tags per sample, then reports evasion detection rate per family.

The MalwareBazaar dataset is dominated by Gootloader/Gootkit (ZIP-based JS
loader). Other families (Vidar, ACRStealer, NetSupport, etc.) are present
but use different delivery mechanisms — their ZIPs are not evasion-based.

Outputs:
  paper/figures/csv/table_family_prevalence.csv
  paper/figures/png/fig11_family_prevalence.png
  paper/figures/pdf/fig11_family_prevalence.pdf
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

CSV_DIR = "paper/figures/csv"
PNG_DIR = "paper/figures/png"
PDF_DIR = "paper/figures/pdf"

PRIMARY_BLUE  = "#0D4EA6"
PRIMARY_RED   = "#B22222"
SUCCESS_GREEN = "#2D6A4F"
AMBER         = "#D4820A"
MED_GRAY      = "#CCCCCC"
DARK_GRAY     = "#444444"

# Known malware family tags to extract (normalised names)
FAMILY_MAP = {
    "gootloader": "Gootloader",
    "gootkit":    "Gootloader",   # same family, different tag
    "Gootloader": "Gootloader",
    "vidar":      "Vidar",
    "Vidar":      "Vidar",
    "acr":        "ACRStealer",
    "ACRStealer": "ACRStealer",
    "netsupport": "NetSupport RAT",
    "NetSupport": "NetSupport RAT",
    "client32":   "NetSupport RAT",
    "salat":      "SalatStealer",
    "SalatStealer": "SalatStealer",
    "smartape":   "SmartApeSG",
    "SmartApeSG": "SmartApeSG",
    "clickfix":   "ClickFix",
    "ClickFix":   "ClickFix",
    "fakecaptcha": "FakeCaptcha",
    "FakeCaptcha": "FakeCaptcha",
    "apt37":      "APT37",
    "APT37":      "APT37",
    "apt36":      "APT36",
    "APT36":      "APT36",
}


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


def extract_primary_family(tags_str: str) -> str:
    """Return the normalised primary malware family from a comma-separated tags string."""
    if pd.isna(tags_str):
        return "Unknown"
    for tag in str(tags_str).split(","):
        tag = tag.strip()
        if tag in FAMILY_MAP:
            return FAMILY_MAP[tag]
    # Fallback: skip IP-like tags, return first non-IP tag
    for tag in str(tags_str).split(","):
        tag = tag.strip()
        if re.match(r"^\d+[\.\-]\d+", tag):
            continue
        if len(tag) < 3:
            continue
        if tag.lower() in ("zip", "ini", "lic"):
            continue
        return tag.capitalize()
    return "Unknown"


def build_family_table() -> pd.DataFrame:
    """Join realworld labels with timestamp tags to get per-family evasion rates."""
    rw = pd.read_csv("data/realworld_labels.csv")
    ts = pd.read_csv("data/bazaar_timestamps.csv")

    # Join key: first 16 chars of filename matches sha256_short in timestamps
    # realworld filename: '02b858299c240c40186d.zip' -> key '02b858299c240c40'
    # timestamps sha256_short: '02b858299c240c40'
    rw["_key"] = rw["filename"].str[:16]
    ts["_key"] = ts["sha256_short"].str[:16]

    merged = rw.merge(ts[["_key", "tags"]], on="_key", how="left")
    merged["family"] = merged["tags"].apply(extract_primary_family)

    # Aggregate per family
    rows = []
    for family, grp in merged.groupby("family"):
        total   = len(grp)
        evasion = int(grp["label"].sum())
        rate    = round(evasion / total * 100, 1) if total > 0 else 0.0
        rows.append({
            "family":          family,
            "samples_scanned": total,
            "evasion_detected": evasion,
            "evasion_rate_pct": rate,
        })

    df = pd.DataFrame(rows).sort_values("samples_scanned", ascending=False).reset_index(drop=True)
    return df


def generate_chart(df: pd.DataFrame):
    """Horizontal bar chart of evasion rate per family."""
    # Only show families with >= 5 samples
    plot_df = df[df["samples_scanned"] >= 5].copy()
    plot_df = plot_df.sort_values("evasion_rate_pct", ascending=True)

    labels = plot_df["family"].tolist()
    rates  = plot_df["evasion_rate_pct"].tolist()
    counts = plot_df["samples_scanned"].tolist()

    colors = [
        SUCCESS_GREEN if r >= 40 else PRIMARY_BLUE if r >= 5 else MED_GRAY
        for r in rates
    ]

    fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(labels) * 0.45)), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Evasion Detection Rate (%)", fontsize=9)
    ax.set_title("Figure 11 — Per-Family Evasion Prevalence (Real-World Scan)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.axvline(x=6.8, color=AMBER, lw=1.2, linestyle="--",
               label="Overall rate (6.8%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4, color=MED_GRAY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate with sample count
    for bar, n, r in zip(bars, counts, rates):
        ax.text(r + 1.0, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=7.5, color=DARK_GRAY)

    _save(fig, "fig11_family_prevalence")


def main():
    print("EXPERIMENT 6 — Per-Family Prevalence Breakdown")
    print("=" * 55)

    configure_style()

    df = build_family_table()

    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = f"{CSV_DIR}/table_family_prevalence.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\nFamily breakdown:")
    print(f"{'Family':<22} {'Scanned':>8} {'Evasion':>8} {'Rate':>8}")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['family']:<22} {row['samples_scanned']:>8} "
              f"{row['evasion_detected']:>8} {row['evasion_rate_pct']:>7.1f}%")

    generate_chart(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
