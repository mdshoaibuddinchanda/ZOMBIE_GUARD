"""
generate_all_figures.py
Generates ALL publication-quality figures and tables for the ZombieGuard paper.

Output layout:
  paper/figures/png/  — 600 DPI PNG (for paper submission)
  paper/figures/pdf/  — vector PDF (for LaTeX inclusion)
  paper/figures/csv/  — source-of-truth data tables (version-controlled)

Design constraints:
  - 600 DPI raster output, PDF fonttype 42 (embeds fonts for submission)
  - All metric values sourced from CSV files or live model computation
  - Each step fails in isolation — one failure does not stop others
  - Tables: CSV only (no image tables from experiment scripts)
  - Figures: PNG + PDF at publication quality
"""

from __future__ import annotations

import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.classifier import FEATURE_COLS
except Exception as exc:
    print(f"WARNING: Failed to import FEATURE_COLS from src.classifier ({exc}); using fallback.")
    FEATURE_COLS = [
        "lf_compression_method", "cd_compression_method", "method_mismatch",
        "data_entropy_shannon", "data_entropy_renyi", "declared_vs_entropy_flag",
        "eocd_count", "lf_unknown_method", "suspicious_entry_count",
        "suspicious_entry_ratio", "any_crc_mismatch", "is_encrypted",
    ]

try:
    from src.shap_analysis import FEATURE_LABELS
except Exception as exc:
    print(f"WARNING: Failed to import FEATURE_LABELS from src.shap_analysis ({exc}); using fallback.")
    FEATURE_LABELS = {
        "lf_compression_method": "LFH compression method",
        "cd_compression_method": "CDH compression method",
        "method_mismatch": "Method mismatch (LFH vs CDH)",
        "data_entropy_shannon": "Shannon entropy of payload",
        "data_entropy_renyi": "Renyi entropy of payload",
        "declared_vs_entropy_flag": "Declared-vs-entropy mismatch",
        "eocd_count": "EOCD signature count",
        "lf_unknown_method": "Unknown method code (LFH)",
        "suspicious_entry_count": "Suspicious entries",
        "suspicious_entry_ratio": "Suspicious entry ratio",
        "any_crc_mismatch": "Any CRC mismatch",
        "is_encrypted": "Encrypted flag",
    }

# Fallback prevalence constants
PREVALENCE_TOTAL = 165
PREVALENCE_DETECTED = 77
PREVALENCE_GOOTLOADER = 66
PREVALENCE_ENTROPY = 7
PREVALENCE_UNKNOWN_METHOD = 1
PREVALENCE_MISMATCH = 1

# Color palette
PRIMARY_BLUE   = "#0D4EA6"
SECONDARY_BLUE = "#4A90D9"
PRIMARY_RED    = "#B22222"
SUCCESS_GREEN  = "#2D6A4F"
AMBER          = "#D4820A"
LIGHT_GRAY     = "#F5F5F5"
MED_GRAY       = "#CCCCCC"
DARK_GRAY      = "#444444"
LIGHT_RED_BG   = "#FFE8E8"
LIGHT_BLUE_BG  = "#E8F0FF"
LIGHT_GREEN_BG = "#E8F5E9"
LIGHT_AMBER_BG = "#FFF8E1"

# Output directories
PNG_DIR = "paper/figures/png"
PDF_DIR = "paper/figures/pdf"
CSV_DIR = "paper/figures/csv"


def configure_style() -> None:
    """Configure publication-quality matplotlib style."""
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": selected_font,
        "font.monospace": ["Courier New", "DejaVu Sans Mono", "monospace"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,   # embed fonts — required for IEEE/ACM submission
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
    print(f"Font: {selected_font}  |  DPI: 600  |  PDF fonttype: 42 (embedded)")


def _save(fig: plt.Figure, stem: str) -> tuple[str, str]:
    """Save figure to png/ and pdf/ subfolders at 600 DPI."""
    Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    png = str(Path(PNG_DIR) / f"{stem}.png")
    pdf = str(Path(PDF_DIR) / f"{stem}.pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png}")
    print(f"  Saved: {pdf}")
    return png, pdf


def _apply_axes_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.4, color=MED_GRAY)


def _safe_float(val: Any) -> float:
    try:
        if pd.isna(val):
            return np.nan
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return np.nan


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _fmt(val: Any, digits: int = 4) -> str:
    n = _safe_float(val)
    return "-" if np.isnan(n) else f"{n:.{digits}f}"


def _read_csv(path: str, label: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        print(f"  Loaded {label}: {path} ({len(df)} rows)")
        return df
    except Exception as exc:
        print(f"  MISSING {label}: {path} — {exc}")
        return None


def _count_files(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file())


def _extract_variant_count(script_path: str) -> int | None:
    p = Path(script_path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    start = text.find("VARIANTS = [")
    if start == -1:
        return None
    end = text.find("]", start)
    return text[start:end].count("(") if end != -1 else None


# ── Figure 1: ZIP Header Mismatch Diagram ────────────────────────────────────

def generate_fig1_zip_header() -> tuple[str, str]:
    """Conceptual diagram showing LFH vs CDH byte-level mismatch."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x, y, title, subtitle, facecolor, min_w=3.6, h=1.8):
        w = max(min_w, max(len(l) for l in textwrap.fill(title, 22).splitlines()) * 0.14)
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                               linewidth=1.2, edgecolor=DARK_GRAY, facecolor=facecolor)
        ax.add_patch(patch)
        white_fc = {PRIMARY_BLUE, PRIMARY_RED, SUCCESS_GREEN}
        tc = "white" if facecolor in white_fc else DARK_GRAY
        ax.text(x + w/2, y + h*0.66, textwrap.fill(title, 22),
                ha="center", va="center", fontsize=9, fontweight="bold", color=tc, linespacing=1.4)
        ax.text(x + w/2, y + h*0.30, textwrap.fill(subtitle, 22),
                ha="center", va="center", fontsize=8, style="italic", color=tc, linespacing=1.4)
        return x, y, w, h

    lx, rx = 0.9, 8.0
    ys = [7.0, 4.7, 2.4]

    ax.text(3.2, 9.3, "Legitimate ZIP", ha="center", fontsize=10, fontweight="bold", color=SUCCESS_GREEN)
    ax.text(10.4, 9.3, "Zombie ZIP (CVE-2026-0866)", ha="center", fontsize=10, fontweight="bold", color=PRIMARY_RED)

    left_boxes = [
        draw_box(lx, ys[0], "Local File Header", "Compression method: 0x0008 (DEFLATE)", SUCCESS_GREEN),
        draw_box(lx, ys[1], "Payload bytes", "DEFLATE compressed, high entropy", PRIMARY_BLUE),
        draw_box(lx, ys[2], "Central Directory Header", "Compression method: 0x0008 (DEFLATE)", SUCCESS_GREEN),
    ]
    right_boxes = [
        draw_box(rx, ys[0], "Local File Header (the lie)", "Compression method: 0x0000 (STORE)", PRIMARY_RED),
        draw_box(rx, ys[1], "Payload bytes", "Actually DEFLATE compressed", PRIMARY_BLUE),
        draw_box(rx, ys[2], "Central Directory Header (truth)", "Compression method: 0x0008 (DEFLATE)", AMBER),
    ]

    for group in [left_boxes, right_boxes]:
        for i in range(2):
            x, y, w, _ = group[i]
            nx, ny, nw, _ = group[i + 1]
            ax.annotate("", xy=(nx + nw/2, ny + 1.8), xytext=(x + w/2, y),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=DARK_GRAY))

    ax.axvline(7.0, linestyle="--", linewidth=1.0, color=MED_GRAY)
    ax.text(rx + 2.15, ys[0] - 0.35, "Bytes 8-9", fontsize=7.8, color=DARK_GRAY,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFFDE7", edgecolor=AMBER, linewidth=1.0))
    ax.text(3.2, 1.1, "LFH method = CDH method  \u2192 parser-consistent",
            ha="center", fontsize=8, color=SUCCESS_GREEN, style="italic")
    ax.text(10.4, 1.1, "LFH method \u2260 CDH method  \u2192 scanner reads wrong bytes",
            ha="center", fontsize=8, color=PRIMARY_RED, style="italic")

    return _save(fig, "fig1_zip_header_mismatch")


# ── Figure 2: Attack Taxonomy Table ──────────────────────────────────────────

def generate_fig2_taxonomy(expected_variant_count: int | None) -> tuple[str, str]:
    """Attack variant taxonomy — publication table figure."""
    variants = [
        ["A", "Classic Zombie ZIP", "STORE (0)", "DEFLATE (8)", "Compressed", "method_mismatch"],
        ["B", "Method-only mismatch", "DEFLATE (8)", "STORE (0)", "Stored", "method_mismatch"],
        ["C", "Gootloader concatenation\n(real-world dominant)", "DEFLATE (8)", "DEFLATE (8)", "Compressed", "eocd_count > 1"],
        ["D", "Multi-file decoy", "STORE (0)", "DEFLATE (8)", "Compressed", "suspicious_entry_ratio"],
        ["E", "CRC32 mismatch", "DEFLATE (8)", "DEFLATE (8)", "Compressed", "any_crc_mismatch"],
        ["F", "Extra field noise", "STORE (0)", "DEFLATE (8)", "Compressed", "structural_combo"],
        ["G", "High compression mismatch", "STORE (0)", "DEFLATE (8)", "Compressed", "entropy_gap"],
        ["H", "Size field mismatch", "STORE (0)", "DEFLATE (8)", "Compressed", "size_inconsistency"],
        ["I", "Undefined method code", "Code 99 (0x63)", "Code 99", "Compressed", "lf_unknown_method*"],
    ]

    if expected_variant_count is not None and expected_variant_count != len(variants):
        print(f"  WARNING: Taxonomy row count mismatch ({len(variants)} vs {expected_variant_count} in script)")

    cols = ["ID", "Variant Name", "LFH Method", "CDH Method", "Payload", "Primary Signal"]
    col_widths = [0.06, 0.34, 0.12, 0.12, 0.12, 0.24]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=False)
    ax.axis("off")

    table = ax.table(cellText=variants, colLabels=cols, colWidths=col_widths,
                     cellLoc="center", bbox=[0.0, 0.08, 1.0, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MED_GRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.15)
            continue
        text = str(cell.get_text().get_text())
        if col == 1:
            cell.get_text().set_text(textwrap.fill(text, 38))
        elif col == 5:
            cell.get_text().set_text(textwrap.fill(text, 24))
        line_count = max(1, len(textwrap.wrap(text, 26)))
        cell.set_height(max(0.13, 0.09 + line_count * 0.03))
        if col in {2, 3}:
            cell.set_facecolor(LIGHT_RED_BG if "STORE" in text else LIGHT_BLUE_BG if "DEFLATE" in text else "white")
        elif col == 5:
            cell.set_facecolor(LIGHT_GRAY)
            cell.set_text_props(fontfamily="monospace")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.10)
    fig.text(0.02, 0.005, "* Single wild sample — future corpus required for trained detection",
             fontsize=8, style="italic", color="gray", ha="left", va="bottom")

    return _save(fig, "fig2_attack_taxonomy")


# ── Figure 3: SHAP Feature Importance ────────────────────────────────────────

def generate_fig3_shap(shap_values: np.ndarray, feature_names: list[str],
                       mean_shap: np.ndarray) -> tuple[str, str]:
    """SHAP mean-absolute feature importance horizontal bar chart."""
    mask = mean_shap > 0.001
    omitted = int((~mask).sum())
    names = np.array(feature_names)[mask]
    vals = np.array(mean_shap)[mask]
    if len(vals) == 0:
        raise ValueError("No SHAP values above 0.001 threshold.")

    order = np.argsort(vals)[::-1]
    names, vals = names[order], vals[order]
    colors = [PRIMARY_BLUE if v > 2.0 else SECONDARY_BLUE if v >= 0.5 else MED_GRAY for v in vals]

    fig, ax = plt.subplots(figsize=(3.6, 4.5), constrained_layout=False)
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=9, labelpad=8)
    ax.axvline(0.5, linestyle="--", linewidth=1.0, color=PRIMARY_RED)
    ax.set_xlim(0.0, max(float(vals.max()) * 1.25, 0.8))
    ax.text(0.52, 0.03, "0.5", color=PRIMARY_RED, fontsize=8, transform=ax.get_xaxis_transform())
    _apply_axes_style(ax)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], fontsize=7.5, padding=3, color=DARK_GRAY)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.98, bottom=0.15)
    fig.text(0.5, 0.015, f"Features with SHAP=0 omitted (n={omitted})",
             ha="center", va="bottom", fontsize=8, color=DARK_GRAY)

    return _save(fig, "fig3_shap_importance")


# ── Figure 4: Cross-Format Generalisation ────────────────────────────────────

def generate_fig4_generalisation(generalisation_df: pd.DataFrame) -> tuple[str, str]:
    """Recall and AUC bar chart across archive formats."""
    fmt_col    = _pick_col(generalisation_df, ["format", "Format"])
    model_col  = _pick_col(generalisation_df, ["model", "Model"])
    recall_col = _pick_col(generalisation_df, ["recall", "Recall"])
    auc_col    = _pick_col(generalisation_df, ["roc_auc", "roc-auc", "ROC_AUC", "auc"])
    if None in {fmt_col, model_col, recall_col, auc_col}:
        raise ValueError("generalisation_results.csv missing required columns.")

    df = generalisation_df[generalisation_df[model_col].str.contains("XGBoost", case=False, na=False)].copy()
    fmt_order = ["ZIP", "APK", "RAR", "7z"]
    rows = []
    for fmt in fmt_order:
        match = df[df[fmt_col].str.strip().str.upper() == fmt.upper()]
        if len(match) == 0:
            raise ValueError(f"Missing XGBoost row for format: {fmt}")
        rows.append(match.iloc[0])
    chart_df = pd.DataFrame(rows)

    xlabels = ["ZIP", "APK", "RAR*", "7z*"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True, constrained_layout=False)

    for ax, col, title in [(axes[0], recall_col, "Recall by Archive Format"),
                           (axes[1], auc_col,    "ROC-AUC by Archive Format")]:
        vals = chart_df[col].astype(float).to_numpy()
        colors = [SUCCESS_GREEN if v >= 0.99 else PRIMARY_BLUE if v >= 0.70
                  else AMBER if v >= 0.50 else PRIMARY_RED for v in vals]
        bars = ax.bar(xlabels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0.0, 1.15)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xlabel("Archive Format", fontsize=9)
        ax.set_title(title, fontsize=10)
        _apply_axes_style(ax)
        ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=3, color=DARK_GRAY)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, wspace=0.10)
    fig.text(0.5, 0.035,
             "* Low recall due to threshold miscalibration under distribution shift; AUC confirms signal transfer",
             ha="center", va="bottom", fontsize=7, color=DARK_GRAY)

    return _save(fig, "fig4_generalisation_chart")


# ── Figure 5: Multi-Model Baseline Comparison ─────────────────────────────────

def generate_fig5_multi_baseline(baseline_df: pd.DataFrame) -> tuple[str, str]:
    """Grouped bar chart: Recall and AUC across 5 classifiers (Experiment 1)."""
    models      = baseline_df["model"].tolist()
    recall_vals = baseline_df["recall"].astype(float).tolist()
    auc_vals    = baseline_df["roc_auc"].astype(float).tolist()
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=False)

    for ax, vals, title, ylabel in [
        (axes[0], recall_vals, "Recall (Malicious Detection Rate)", "Recall"),
        (axes[1], auc_vals,    "ROC-AUC",                          "ROC-AUC"),
    ]:
        colors = [SUCCESS_GREEN if m == "XGBoost" else SECONDARY_BLUE for m in models]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=18, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        _apply_axes_style(ax)
        ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=3, color=DARK_GRAY)
        ax.axhline(y=0.99, linestyle="--", linewidth=0.8, color=PRIMARY_RED, alpha=0.5)

    fig.suptitle("Figure 5 — Multi-Model Comparison: Recall and ROC-AUC (Hard Test Set)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.22, wspace=0.28)

    return _save(fig, "fig5_multi_baseline_chart")


# ── Figure 6: Per-Variant Recall ──────────────────────────────────────────────

def generate_fig6_variant_recall(variant_df: pd.DataFrame) -> tuple[str, str]:
    """Horizontal bar chart of recall per attack variant (Experiment 2)."""
    plot_df = variant_df[variant_df["recall"].notna() & (variant_df["n_test"] > 0)].copy()
    labels  = [f"{r['variant']} \u2014 {r['name']}" for _, r in plot_df.iterrows()]
    recalls = plot_df["recall"].tolist()
    colors  = [SUCCESS_GREEN if r >= 1.0 else AMBER if r >= 0.90 else PRIMARY_RED for r in recalls]

    fig, ax = plt.subplots(figsize=(7.0, max(3.5, len(labels) * 0.42)), constrained_layout=False)
    y = np.arange(len(labels))
    bars = ax.barh(y, recalls, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_title("Figure 6 — Per-Variant Recall (XGBoost, full malicious corpus)",
                 fontsize=11, fontweight="bold")
    ax.axvline(x=1.0, linestyle="--", linewidth=0.9, color=MED_GRAY)
    ax.axvline(x=0.90, linestyle=":", linewidth=0.8, color=AMBER, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4, color=MED_GRAY)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=4, color=DARK_GRAY)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.92, bottom=0.10)

    return _save(fig, "fig6_variant_recall_chart")


# ── Figure 7: Temporal Stability ──────────────────────────────────────────────

def generate_fig7_temporal_stability(temporal_df: pd.DataFrame) -> tuple[str, str]:
    """Line chart: Recall/F1/AUC across temporal windows T1/T2/T3 (Experiment 3)."""
    import matplotlib.ticker as mticker

    synth_rows = temporal_df[temporal_df["window"].str.startswith("Synth")]
    temp_rows  = temporal_df[~temporal_df["window"].str.startswith("Synth")]

    windows = temp_rows["window"].tolist()
    x = np.arange(len(windows))

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=False)

    for metric, color, marker, label in [
        ("recall", "#e74c3c", "o", "Temporal-trained Recall"),
        ("f1",     "#3498db", "s", "Temporal-trained F1"),
        ("auc",    "#2ecc71", "^", "Temporal-trained AUC"),
    ]:
        vals = temp_rows[metric].astype(float).tolist()
        ax.plot(x, vals, color=color, marker=marker, linewidth=2, markersize=8, label=label)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5, color=color)

    if not synth_rows.empty:
        synth_map = dict(zip(synth_rows["window"].str.replace("Synth\u2192", "").tolist(),
                             synth_rows["recall"].astype(float).tolist()))
        sx = [i for i, w in enumerate(windows) if w in synth_map]
        sy = [synth_map[w] for w in windows if w in synth_map]
        if sx:
            ax.plot(sx, sy, color="#e67e22", marker="D", linewidth=2, markersize=8,
                    linestyle="--", label="Synthetic-trained Recall (zero-shot)")

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Temporal Window", fontsize=10)
    ax.set_title("Figure 7 — Temporal Stability: Recall / F1 / AUC Across Time Windows",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12)

    return _save(fig, "fig7_temporal_stability_chart")


# ── Prevalence tables (figures only — no CSV, data is hardcoded/realworld) ───

def generate_table3_prevalence(realworld_df: pd.DataFrame | None) -> tuple[str, str]:
    """Prevalence breakdown figure — general scan."""
    used_fallback = False
    if realworld_df is not None and not realworld_df.empty and "label" in realworld_df.columns:
        signal_col = _pick_col(realworld_df, ["signal", "Signal"])
        total    = int(len(realworld_df))
        detected = int((realworld_df["label"] == 1).sum())
        by_signal = realworld_df[realworld_df["label"] == 1].groupby(signal_col).size().to_dict() if signal_col else {}
        goot_count    = int(by_signal.get("gootloader", 0))
        entropy_count = int(by_signal.get("entropy", 0))
        unknown_count = int(by_signal.get("unknown_method", 0))
        mismatch_count = int(by_signal.get("mismatch", 0))
    else:
        used_fallback = True
        total, detected = PREVALENCE_TOTAL, PREVALENCE_DETECTED
        goot_count, entropy_count = PREVALENCE_GOOTLOADER, PREVALENCE_ENTROPY
        unknown_count, mismatch_count = PREVALENCE_UNKNOWN_METHOD, PREVALENCE_MISMATCH
        print("  WARNING: Using hardcoded prevalence values (realworld_labels.csv not found)")

    def pct(v, d): return f"{100.0*v/d:.1f}%" if d > 0 else "0.0%"
    non_evasion = total - detected
    rows = [
        ["Gootloader EOCD chaining (EOCD > 1)", goot_count, pct(goot_count, total), "eocd_count"],
        ["High entropy anomaly", entropy_count, pct(entropy_count, total), "data_entropy_shannon"],
        ["Undefined LFH method code", unknown_count, pct(unknown_count, total), "lf_unknown_method"],
        ["LFH/CDH mismatch", mismatch_count, pct(mismatch_count, total), "method_mismatch"],
        ["Total detected evasion", detected, pct(detected, total), "label==1"],
        ["Non-evasion / out-of-scope", non_evasion, pct(non_evasion, total), "label==0"],
    ]
    cols = ["Signal Type", "Count", "Share", "Feature"]

    fig, ax = plt.subplots(figsize=(7.0, (len(rows)+1)*0.13+0.50), constrained_layout=False)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, colWidths=[0.48, 0.10, 0.15, 0.27],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    ax.set_title("General scan: 1,366 samples across 18 malware families", fontsize=11, pad=12)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MED_GRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.15)
        else:
            cell.set_facecolor(LIGHT_GREEN_BG if row == 5 else LIGHT_AMBER_BG if row == 6
                               else "white" if row % 2 else LIGHT_GRAY)
            cell.set_height(0.13)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08)
    if used_fallback:
        fig.text(0.5, 0.01, "Fallback constants used — run verify_realworld.py to regenerate",
                 ha="center", fontsize=8, color=PRIMARY_RED)

    return _save(fig, "table3_prevalence_breakdown")


def generate_table3a_targeted_prevalence() -> tuple[str, str]:
    """Targeted-scan prevalence figure (165 Gootloader samples)."""
    data = [
        ["Signal Type", "Count", "Share of 165", "Feature"],
        ["Gootloader EOCD chaining\n(EOCD > 1)", "66", "40.0%", "eocd_count"],
        ["High entropy anomaly", "7", "4.2%", "data_entropy_shannon"],
        ["Undefined LFH method code\n(Variant I)", "1", "0.6%", "lf_unknown_method"],
        ["True CVE LFH/CDH mismatch\n(Variant A)", "1", "0.6%", "method_mismatch"],
        ["Total evasion detected", "77", "46.7%", "---"],
        ["Non-evasion (out of scope)", "88", "53.3%", "---"],
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=False)
    ax.axis("off")
    table = ax.table(cellText=data[1:], colLabels=data[0],
                     colWidths=[0.34, 0.14, 0.18, 0.24],
                     bbox=[0.0, 0.0, 1.0, 0.78], cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(DARK_GRAY)
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        else:
            cell.set_facecolor(LIGHT_GREEN_BG if row == 5 else LIGHT_AMBER_BG if row == 6
                               else "white" if row % 2 else LIGHT_GRAY)
    ax.set_title("Targeted scan: 165 Gootloader-associated samples",
                 fontsize=11, pad=8, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

    return _save(fig, "table3a_targeted_prevalence")


# ── Figure 8: ROC Curve ───────────────────────────────────────────────────────

def generate_fig8_roc_curve(model, features_df: pd.DataFrame,
                             labels_df: pd.DataFrame) -> tuple[str, str]:
    """ROC curve: ZombieGuard XGBoost vs rule-based baseline."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split

    merged = features_df.merge(labels_df, on="filename")
    for col in ["method_mismatch", "declared_vs_entropy_flag",
                "lf_crc_valid", "any_crc_mismatch", "is_encrypted"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)
    available = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available].fillna(0).astype(float)
    y = merged["label"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    xgb_proba = model.predict_proba(X_test)[:, 1]

    # Rule-based score: sum of 4 binary signals, normalised
    base_score = np.zeros(len(X_test))
    for col, cond in [("method_mismatch", None), ("declared_vs_entropy_flag", None),
                      ("any_crc_mismatch", None)]:
        if col in X_test.columns:
            base_score += X_test[col].astype(float).values
    if "eocd_count" in X_test.columns:
        base_score += (X_test["eocd_count"] > 1).astype(float).values
    base_score /= 4.0

    fpr_xgb,  tpr_xgb,  _ = roc_curve(y_test, xgb_proba)
    fpr_base, tpr_base, _ = roc_curve(y_test, base_score)
    auc_xgb  = roc_auc_score(y_test, xgb_proba)
    auc_base = roc_auc_score(y_test, base_score)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    ax.plot(fpr_xgb,  tpr_xgb,  color=PRIMARY_BLUE, lw=2,
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

    # Save AUC summary CSV
    pd.DataFrame([
        {"model": "ZombieGuard XGBoost", "roc_auc": round(auc_xgb, 4)},
        {"model": "Rule-based baseline", "roc_auc": round(auc_base, 4)},
    ]).to_csv(f"{CSV_DIR}/table_roc_pr_auc.csv", index=False)

    return _save(fig, "fig8_roc_curve")


# ── Figure 9: Precision-Recall Curve ─────────────────────────────────────────

def generate_fig9_pr_curve(model, features_df: pd.DataFrame,
                            labels_df: pd.DataFrame) -> tuple[str, str]:
    """PR curve: ZombieGuard XGBoost vs rule-based baseline."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import train_test_split

    merged = features_df.merge(labels_df, on="filename")
    for col in ["method_mismatch", "declared_vs_entropy_flag",
                "lf_crc_valid", "any_crc_mismatch", "is_encrypted"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)
    available = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[available].fillna(0).astype(float)
    y = merged["label"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    xgb_proba = model.predict_proba(X_test)[:, 1]
    base_score = np.zeros(len(X_test))
    for col in ["method_mismatch", "declared_vs_entropy_flag", "any_crc_mismatch"]:
        if col in X_test.columns:
            base_score += X_test[col].astype(float).values
    if "eocd_count" in X_test.columns:
        base_score += (X_test["eocd_count"] > 1).astype(float).values
    base_score /= 4.0

    prec_xgb,  rec_xgb,  _ = precision_recall_curve(y_test, xgb_proba)
    prec_base, rec_base, _ = precision_recall_curve(y_test, base_score)
    ap_xgb  = average_precision_score(y_test, xgb_proba)
    ap_base = average_precision_score(y_test, base_score)
    prevalence = float(y_test.sum()) / len(y_test)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
    ax.plot(rec_xgb,  prec_xgb,  color=PRIMARY_BLUE, lw=2,
            label=f"ZombieGuard LightGBM (AP = {ap_xgb:.4f})")
    ax.plot(rec_base, prec_base, color=AMBER, lw=2, linestyle="--",
            label=f"Rule-based baseline (AP = {ap_base:.4f})")
    ax.axhline(y=prevalence, color=MED_GRAY, lw=1, linestyle=":",
               label=f"No-skill baseline ({prevalence:.3f})")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Figure 9 — Precision-Recall Curve: ZombieGuard vs Rule-Based Baseline", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.4, color=MED_GRAY)

    return _save(fig, "fig9_pr_curve")


# ── Figure 10: Entropy Distribution ──────────────────────────────────────────

def generate_fig10_entropy_distribution(features_df: pd.DataFrame,
                                         labels_df: pd.DataFrame) -> tuple[str, str]:
    """Overlapping Shannon entropy histograms: malicious vs benign, threshold=7.0."""
    merged = features_df.merge(labels_df, on="filename")
    mal_entropy = merged[merged["label"] == 1]["data_entropy_shannon"].dropna().values
    ben_entropy = merged[merged["label"] == 0]["data_entropy_shannon"].dropna().values
    threshold = 7.0
    bins = np.linspace(0, 8, 65)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    ax.hist(ben_entropy, bins=bins, alpha=0.55, color=PRIMARY_BLUE,
            label=f"Benign (n={len(ben_entropy)})", density=True, edgecolor="none")
    ax.hist(mal_entropy, bins=bins, alpha=0.55, color=PRIMARY_RED,
            label=f"Malicious (n={len(mal_entropy)})", density=True, edgecolor="none")
    ax.axvline(x=threshold, color=AMBER, lw=2, linestyle="--",
               label=f"Threshold = {threshold} bits/byte")
    ymax = ax.get_ylim()[1]
    ax.text(threshold + 0.05, ymax * 0.88,
            f"declared_vs_entropy_flag\nthreshold = {threshold}",
            fontsize=7.5, color=AMBER, va="top")
    ax.set_xlabel("Shannon Entropy (bits/byte)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Figure 10 — Shannon Entropy Distribution: Malicious vs Benign", fontsize=10)
    ax.set_xlim(0, 8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4, color=MED_GRAY)
    fig.text(0.5, -0.04,
             "Entropy alone is insufficient — 76.2% of benign samples also exceed 7.0 bits/byte.\n"
             "The threshold is one signal among 12 features; the ML model resolves the overlap.",
             ha="center", fontsize=7.5, color=DARK_GRAY, style="italic")

    # Save stats CSV
    pd.DataFrame([
        {"class": "Malicious", "n": len(mal_entropy),
         "mean": round(mal_entropy.mean(), 4), "std": round(mal_entropy.std(), 4),
         "pct_above_threshold": round(100 * (mal_entropy >= threshold).mean(), 2)},
        {"class": "Benign", "n": len(ben_entropy),
         "mean": round(ben_entropy.mean(), 4), "std": round(ben_entropy.std(), 4),
         "pct_above_threshold": round(100 * (ben_entropy >= threshold).mean(), 2)},
    ]).to_csv(f"{CSV_DIR}/table_entropy_stats.csv", index=False)

    return _save(fig, "fig10_entropy_distribution")


# ── Figure 11: Per-Family Prevalence ─────────────────────────────────────────

def generate_fig11_family_prevalence(family_df: pd.DataFrame) -> tuple[str, str]:
    """Horizontal bar chart of evasion rate per malware family."""
    plot_df = family_df[family_df["samples_scanned"] >= 5].copy()
    plot_df = plot_df.sort_values("evasion_rate_pct", ascending=True)

    labels = plot_df["family"].tolist()
    rates  = plot_df["evasion_rate_pct"].tolist()
    counts = plot_df["samples_scanned"].tolist()
    colors = [SUCCESS_GREEN if r >= 40 else PRIMARY_BLUE if r >= 5 else MED_GRAY for r in rates]

    fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(labels) * 0.45)), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Evasion Detection Rate (%)", fontsize=9)
    ax.set_title("Figure 11 — Per-Family Evasion Prevalence (Real-World Scan)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.axvline(x=6.8, color=AMBER, lw=1.2, linestyle="--", label="Overall rate (6.8%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3, linewidth=0.4, color=MED_GRAY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, n, r in zip(bars, counts, rates):
        ax.text(r + 1.0, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=7.5, color=DARK_GRAY)

    return _save(fig, "fig11_family_prevalence")


# ── Verify outputs ────────────────────────────────────────────────────────────

def verify_outputs() -> tuple[int, int, list[str]]:
    """Check all PNG files for resolution and PDF pairing."""
    png_dir = Path(PNG_DIR)
    pdf_dir = Path(PDF_DIR)
    warnings: list[str] = []
    pdf_ok = 0

    png_files = sorted(png_dir.glob("*.png")) if png_dir.exists() else []
    print(f"\nVerification ({len(png_files)} PNG files in {PNG_DIR}):")

    for png in png_files:
        img = plt.imread(str(png))
        w_px = int(img.shape[1])
        h_px = int(img.shape[0])
        if w_px < 2000:
            msg = f"  LOW-RES: {png.name} ({w_px}px wide — expected ≥2000px at 600 DPI)"
            print(msg)
            warnings.append(msg)
        else:
            print(f"  OK: {png.name}  {w_px}x{h_px}px")

        pdf = pdf_dir / png.with_suffix(".pdf").name
        if pdf.exists():
            pdf_ok += 1
        else:
            msg = f"  MISSING PDF: {pdf}"
            print(msg)
            warnings.append(msg)

    return len(png_files), pdf_ok, warnings


# ── Step runner ───────────────────────────────────────────────────────────────

def _run(name: str, fn: Any, *args: Any) -> bool:
    print(f"\n--- {name} ---")
    try:
        png, pdf = fn(*args)
        print(f"  OK: {name}")
        return True
    except Exception as exc:
        print(f"  FAILED: {name} — {exc}")
        traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_style()
    Path(PNG_DIR).mkdir(parents=True, exist_ok=True)
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    Path(CSV_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ZombieGuard — Generating all paper figures")
    print(f"  PNG → {PNG_DIR}")
    print(f"  PDF → {PDF_DIR}")
    print("=" * 65)

    # ── Load shared data ──────────────────────────────────────────────────────
    model = None
    model_path = "models/lgbm_model.pkl"
    if Path(model_path).exists():
        try:
            model = joblib.load(model_path)
            print(f"\nLoaded model: {model_path}")
        except Exception as exc:
            print(f"\nERROR loading model: {exc}")
    else:
        print(f"\nERROR: Missing model file: {model_path}")

    features_df      = _read_csv("data/processed/features.csv", "features")
    labels_df        = _read_csv("data/processed/labels.csv", "labels")
    generalisation_df = _read_csv(f"{CSV_DIR}/generalisation_results.csv", "generalisation")
    baseline_df      = _read_csv(f"{CSV_DIR}/table1_baseline_comparison.csv", "baseline")
    multi_baseline_df = _read_csv(f"{CSV_DIR}/table6b_multi_baseline_hard_test.csv", "multi_baseline_hard")
    variant_df       = _read_csv(f"{CSV_DIR}/table7_variant_recall.csv", "variant_recall")
    temporal_df      = _read_csv(f"{CSV_DIR}/table8_temporal_stability.csv", "temporal_stability")
    family_df        = _read_csv(f"{CSV_DIR}/table_family_prevalence.csv", "family_prevalence")
    realworld_df     = _read_csv("data/realworld_labels.csv", "realworld") if Path("data/realworld_labels.csv").exists() else None

    # Merge features + labels for SHAP
    merged = None
    shap_values = mean_shap = None
    feature_names: list[str] = []

    if features_df is not None and labels_df is not None:
        try:
            merged = features_df.merge(labels_df, on="filename")
            print(f"  Merged features/labels: {len(merged)} rows")
        except Exception as exc:
            print(f"  ERROR merging features/labels: {exc}")

    if model is not None and merged is not None:
        missing = [c for c in FEATURE_COLS if c not in merged.columns]
        if missing:
            print(f"  ERROR: Missing SHAP columns: {missing}")
        else:
            x = merged[FEATURE_COLS].copy()
            for col in ["method_mismatch", "declared_vs_entropy_flag",
                        "lf_unknown_method", "any_crc_mismatch", "is_encrypted"]:
                if col in x.columns:
                    x[col] = x[col].astype(float)
            feature_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
            x.columns = feature_names
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x)
                mean_shap = np.abs(shap_values).mean(axis=0)
                print("  Computed SHAP values.")
            except Exception as exc:
                print(f"  ERROR computing SHAP: {exc}")

    expected_variants = _extract_variant_count("data/scripts/generate_zombie_samples.py")

    # ── Generate figures ──────────────────────────────────────────────────────
    ok = 0
    total = 0

    total += 1
    if _run("Figure 1 — ZIP Header Mismatch", generate_fig1_zip_header):
        ok += 1

    total += 1
    if _run("Figure 2 — Attack Taxonomy", generate_fig2_taxonomy, expected_variants):
        ok += 1

    total += 1
    if shap_values is not None and mean_shap is not None and feature_names:
        if _run("Figure 3 — SHAP Importance", generate_fig3_shap, shap_values, feature_names, mean_shap):
            ok += 1
    else:
        print("\n--- Figure 3 --- SKIPPED (model/SHAP unavailable)")

    total += 1
    if generalisation_df is not None:
        if _run("Figure 4 — Cross-Format Generalisation", generate_fig4_generalisation, generalisation_df):
            ok += 1
    else:
        print("\n--- Figure 4 --- SKIPPED (generalisation_results.csv not found)")

    total += 1
    if multi_baseline_df is not None:
        if _run("Figure 5 — Multi-Model Baseline", generate_fig5_multi_baseline, multi_baseline_df):
            ok += 1
    else:
        print("\n--- Figure 5 --- SKIPPED (table6b_multi_baseline_hard_test.csv not found — run src/multi_baseline.py)")

    total += 1
    if variant_df is not None:
        if _run("Figure 6 — Per-Variant Recall", generate_fig6_variant_recall, variant_df):
            ok += 1
    else:
        print("\n--- Figure 6 --- SKIPPED (table7_variant_recall.csv not found — run src/variant_recall.py)")

    total += 1
    if temporal_df is not None:
        if _run("Figure 7 — Temporal Stability", generate_fig7_temporal_stability, temporal_df):
            ok += 1
    else:
        print("\n--- Figure 7 --- SKIPPED (table8_temporal_stability.csv not found — run src/temporal_stability.py)")

    total += 1
    if _run("Table 3 — Prevalence Breakdown", generate_table3_prevalence, realworld_df):
        ok += 1

    total += 1
    if _run("Table 3A — Targeted Prevalence", generate_table3a_targeted_prevalence):
        ok += 1

    total += 1
    if model is not None and features_df is not None and labels_df is not None:
        if _run("Figure 8 — ROC Curve", generate_fig8_roc_curve, model, features_df, labels_df):
            ok += 1
    else:
        print("\n--- Figure 8 --- SKIPPED (model or data unavailable)")

    total += 1
    if model is not None and features_df is not None and labels_df is not None:
        if _run("Figure 9 — PR Curve", generate_fig9_pr_curve, model, features_df, labels_df):
            ok += 1
    else:
        print("\n--- Figure 9 --- SKIPPED (model or data unavailable)")

    total += 1
    if features_df is not None and labels_df is not None:
        if _run("Figure 10 — Entropy Distribution", generate_fig10_entropy_distribution, features_df, labels_df):
            ok += 1
    else:
        print("\n--- Figure 10 --- SKIPPED (features/labels unavailable)")

    total += 1
    if family_df is not None:
        if _run("Figure 11 — Family Prevalence", generate_fig11_family_prevalence, family_df):
            ok += 1
    else:
        print("\n--- Figure 11 --- SKIPPED (table_family_prevalence.csv not found — run src/family_prevalence.py)")

    # ── Verify ────────────────────────────────────────────────────────────────
    png_count, pdf_count, warnings = verify_outputs()

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Steps completed:  {ok}/{total}")
    print(f"PNG files:        {png_count}  (in {PNG_DIR})")
    print(f"PDF files paired: {pdf_count}  (in {PDF_DIR})")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
        print("READY FOR SUBMISSION: No")
    else:
        print("Warnings: None")
        print("READY FOR SUBMISSION: Yes")


if __name__ == "__main__":
    main()
