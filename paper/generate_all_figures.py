"""
generate_all_figures.py
Generates publication-quality figures and tables for the ZombieGuard paper.

Design constraints:
- All metric values must come from CSV/model/SHAP runtime computation.
- Each output is saved as 600 DPI PNG and PDF.
- Failures are isolated so one figure failure does not stop others.
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

# Try importing shared feature configs. Fall back if import fails.
try:
    from src.classifier import FEATURE_COLS
except Exception as exc:
    print(
        "WARNING: Failed to import FEATURE_COLS from src.classifier; "
        f"using fallback. ({exc})"
    )
    FEATURE_COLS = [
        "lf_compression_method",
        "cd_compression_method",
        "method_mismatch",
        "data_entropy_shannon",
        "data_entropy_renyi",
        "declared_vs_entropy_flag",
        "eocd_count",
        "lf_unknown_method",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
        "any_crc_mismatch",
        "is_encrypted",
    ]

try:
    from src.shap_analysis import FEATURE_LABELS
except Exception as exc:
    print(
        "WARNING: Failed to import FEATURE_LABELS from src.shap_analysis; "
        f"using fallback. ({exc})"
    )
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


# Prevalence fallback constants (allowed fallback path only).
PREVALENCE_TOTAL = 165
PREVALENCE_DETECTED = 77
PREVALENCE_GOOTLOADER = 66
PREVALENCE_ENTROPY = 7
PREVALENCE_UNKNOWN_METHOD = 1
PREVALENCE_MISMATCH = 1

PRIMARY_BLUE = "#0D4EA6"
SECONDARY_BLUE = "#4A90D9"
PRIMARY_RED = "#B22222"
SUCCESS_GREEN = "#2D6A4F"
AMBER = "#D4820A"
LIGHT_GRAY = "#F5F5F5"
MED_GRAY = "#CCCCCC"
DARK_GRAY = "#444444"
LIGHT_RED_BG = "#FFE8E8"
LIGHT_BLUE_BG = "#E8F0FF"
LIGHT_GREEN_BG = "#E8F5E9"
LIGHT_AMBER_BG = "#FFF8E1"


def configure_style() -> None:
    """Configure publication style and font embedding settings."""
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_font = (
        "Times New Roman" if "Times New Roman" in available_fonts
        else "DejaVu Serif"
    )

    plt.rcParams.update({
        "font.family": selected_font,
        "font.monospace": ["Courier New", "DejaVu Sans Mono", "monospace"],
        "font.size": 9,
        "axes.titlesize": 10,
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
        "figure.constrained_layout.use": True,
    })
    print(f"Using font family: {selected_font}")


def _save_png_pdf(fig: plt.Figure, output_dir: str, stem: str) -> tuple[str, str]:
    """Save figure as both 600-DPI PNG and PDF."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path), str(pdf_path)


def _apply_axes_style(ax: plt.Axes) -> None:
    """Apply consistent axis and grid styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.4, color=MED_GRAY)


def _safe_float(val: Any) -> float:
    """Convert value to float; return nan on failure."""
    try:
        if pd.isna(val):
            return np.nan
        return float(val)
    except Exception:
        return np.nan


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first existing column from a candidate list."""
    cols = set(df.columns)
    for col in candidates:
        if col in cols:
            return col
    return None


def _format_metric(val: Any, digits: int = 4) -> str:
    """Format metric values consistently for table display."""
    num = _safe_float(val)
    if np.isnan(num):
        return "-"
    return f"{num:.{digits}f}"


def _read_csv_or_none(path: str, label: str) -> pd.DataFrame | None:
    """Read a CSV safely; print clear error and return None on failure."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded {label}: {path} (rows={len(df)})")
        return df
    except Exception as exc:
        print(f"ERROR: Missing or unreadable {label} at {path}: {exc}")
        print(f"Skipping outputs that depend on {label}.")
        return None


def _count_files(path: str) -> int | None:
    """Count files in a directory recursively; return None if unavailable."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return None
    return sum(1 for child in p.rglob("*") if child.is_file())


def _extract_variant_count(generate_script_path: str) -> int | None:
    """Best-effort extraction of variant count from generator script text."""
    p = Path(generate_script_path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    start = text.find("VARIANTS = [")
    if start == -1:
        return None
    end = text.find("]", start)
    if end == -1:
        return None
    block = text[start:end]
    return block.count("(")


def _check_text_bbox_overlap(
    fig: plt.Figure,
    text_artists: list[Any],
    label: str,
) -> None:
    """Check text bounding box intersections and print warnings."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = 0

    for i in range(len(text_artists)):
        for j in range(i + 1, len(text_artists)):
            bb1 = text_artists[i].get_window_extent(renderer=renderer)
            bb2 = text_artists[j].get_window_extent(renderer=renderer)
            if bb1.overlaps(bb2):
                overlaps += 1

    if overlaps > 0:
        print(
            f"WARNING: {label} has {overlaps} overlapping text bounding boxes."
        )
    else:
        print(f"OK: {label} text bounding boxes do not intersect.")


def generate_fig1_zip_header(output_dir: str) -> tuple[str, str]:
    """Generate byte-level ZIP header mismatch conceptual diagram."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    texts = []

    def draw_box(
        x: float,
        y: float,
        title: str,
        subtitle: str,
        facecolor: str,
        min_w: float = 3.6,
        h: float = 1.8,
    ) -> tuple[float, float, float, float]:
        wrapped_title = textwrap.fill(title, width=22)
        wrapped_subtitle = textwrap.fill(subtitle, width=22)
        longest = max(len(line) for line in wrapped_title.splitlines())
        w = max(min_w, longest * 0.14)

        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            linewidth=1.2,
            edgecolor=DARK_GRAY,
            facecolor=facecolor,
        )
        ax.add_patch(patch)

        txt1 = ax.text(
            x + w / 2,
            y + h * 0.66,
            wrapped_title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if facecolor in {PRIMARY_BLUE, PRIMARY_RED, SUCCESS_GREEN}
            else DARK_GRAY,
            linespacing=1.4,
        )
        txt2 = ax.text(
            x + w / 2,
            y + h * 0.30,
            wrapped_subtitle,
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
            color="white" if facecolor in {PRIMARY_BLUE, PRIMARY_RED, SUCCESS_GREEN}
            else DARK_GRAY,
            linespacing=1.4,
        )
        texts.extend([txt1, txt2])
        return x, y, w, h

    left_x = 0.9
    right_x = 8.0
    y_positions = [7.0, 4.7, 2.4]

    left_title = ax.text(
        3.2,
        9.3,
        "Legitimate ZIP",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=SUCCESS_GREEN,
    )
    right_title = ax.text(
        10.4,
        9.3,
        "Zombie ZIP (CVE-2026-0866)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=PRIMARY_RED,
    )
    texts.extend([left_title, right_title])

    left_boxes = [
        draw_box(
            left_x,
            y_positions[0],
            "Local File Header",
            "Compression method: 0x0008 (DEFLATE)",
            SUCCESS_GREEN,
        ),
        draw_box(
            left_x,
            y_positions[1],
            "Payload bytes",
            "DEFLATE compressed, high entropy",
            PRIMARY_BLUE,
        ),
        draw_box(
            left_x,
            y_positions[2],
            "Central Directory Header",
            "Compression method: 0x0008 (DEFLATE)",
            SUCCESS_GREEN,
        ),
    ]

    right_boxes = [
        draw_box(
            right_x,
            y_positions[0],
            "Local File Header (the lie)",
            "Compression method: 0x0000 (STORE)",
            PRIMARY_RED,
        ),
        draw_box(
            right_x,
            y_positions[1],
            "Payload bytes",
            "Actually DEFLATE compressed",
            PRIMARY_BLUE,
        ),
        draw_box(
            right_x,
            y_positions[2],
            "Central Directory Header (truth)",
            "Compression method: 0x0008 (DEFLATE)",
            AMBER,
        ),
    ]

    for group in [left_boxes, right_boxes]:
        for idx in range(2):
            x, y, w, h = group[idx]
            nx, ny, nw, _ = group[idx + 1]
            ax.annotate(
                "",
                xy=(nx + nw / 2, ny + 1.8),
                xytext=(x + w / 2, y),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=DARK_GRAY),
            )

    ax.axvline(7.0, linestyle="--", linewidth=1.0, color=MED_GRAY)

    callout = ax.text(
        right_x + 0.45,
        y_positions[0] + 1.95,
        "Bytes 8-9",
        fontsize=8,
        color=DARK_GRAY,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FFFDE7",
            edgecolor=AMBER,
            linewidth=1.0,
        ),
    )
    texts.append(callout)

    s_left = ax.text(
        3.2,
        1.1,
        "LFH method = CDH method  -> parser-consistent",
        ha="center",
        va="center",
        fontsize=8,
        color=SUCCESS_GREEN,
        style="italic",
    )
    s_right = ax.text(
        10.4,
        1.1,
        "LFH method != CDH method  -> scanner reads wrong bytes",
        ha="center",
        va="center",
        fontsize=8,
        color=PRIMARY_RED,
        style="italic",
    )
    texts.extend([s_left, s_right])

    _check_text_bbox_overlap(fig, texts, "Figure 1")

    return _save_png_pdf(fig, output_dir, "fig1_zip_header_mismatch")


def generate_fig2_taxonomy(
    output_dir: str,
    expected_variant_count: int | None,
) -> tuple[str, str]:
    """Generate taxonomy table for archive evasion variants."""
    variants = [
        ["A", "Classic Zombie ZIP", "STORE (0)", "DEFLATE (8)", "Compressed", "method_mismatch"],
        ["B", "Method-only mismatch", "DEFLATE (8)", "STORE (0)", "Stored", "method_mismatch"],
        ["C", "Gootloader concatenation (real-world dominant)", "DEFLATE (8)", "DEFLATE (8)", "Compressed", "eocd_count > 1"],
        ["D", "Multi-file decoy", "STORE (0)", "DEFLATE (8)", "Compressed", "suspicious_entry_ratio"],
        ["E", "CRC32 mismatch", "DEFLATE (8)", "DEFLATE (8)", "Compressed", "any_crc_mismatch"],
        ["F", "Extra field noise", "STORE (0)", "DEFLATE (8)", "Compressed", "structural_combo"],
        ["G", "High compression mismatch", "STORE (0)", "DEFLATE (8)", "Compressed", "entropy_gap"],
        ["H", "Size field mismatch", "STORE (0)", "DEFLATE (8)", "Compressed", "size_inconsistency"],
    ]

    if expected_variant_count is not None and expected_variant_count != len(variants):
        print(
            "WARNING: Taxonomy row count does not match VARIANTS in "
            f"generate_zombie_samples.py ({len(variants)} vs {expected_variant_count})."
        )

    cols = ["ID", "Variant Name", "LFH Method", "CDH Method", "Payload", "Primary Signal"]
    col_widths = [0.07, 0.31, 0.14, 0.14, 0.13, 0.21]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=variants,
        colLabels=cols,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
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
        line_count = max(1, len(textwrap.wrap(text, width=26)))
        row_h = max(0.13, 0.09 + line_count * 0.03)
        cell.set_height(row_h)

        if col in {2, 3}:
            if "STORE" in text:
                cell.set_facecolor(LIGHT_RED_BG)
            elif "DEFLATE" in text:
                cell.set_facecolor(LIGHT_BLUE_BG)
            else:
                cell.set_facecolor("white")
        elif col == 5:
            cell.set_facecolor(LIGHT_GRAY)
            cell.set_text_props(fontfamily="monospace")
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)

    return _save_png_pdf(fig, output_dir, "fig2_attack_taxonomy")


def generate_fig3_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    mean_shap: np.ndarray,
    output_dir: str,
) -> tuple[str, str]:
    """Generate SHAP mean-absolute feature importance chart."""
    if shap_values is None or mean_shap is None or len(feature_names) == 0:
        raise ValueError("Missing SHAP arrays or feature names.")

    included_mask = mean_shap > 0.001
    omitted = int((~included_mask).sum())

    filt_names = np.array(feature_names)[included_mask]
    filt_vals = np.array(mean_shap)[included_mask]
    if len(filt_vals) == 0:
        raise ValueError("No SHAP values above 0.001 threshold.")

    order = np.argsort(filt_vals)[::-1]
    names_sorted = filt_names[order]
    vals_sorted = filt_vals[order]

    colors = []
    for val in vals_sorted:
        if val > 2.0:
            colors.append(PRIMARY_BLUE)
        elif val >= 0.5:
            colors.append(SECONDARY_BLUE)
        else:
            colors.append(MED_GRAY)

    fig, ax = plt.subplots(figsize=(3.5, 4.2), constrained_layout=True)
    bars = ax.barh(range(len(vals_sorted)), vals_sorted, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("Figure 3. SHAP Feature Importance", fontsize=10)

    ax.axvline(0.5, linestyle="--", linewidth=1.0, color=PRIMARY_RED)
    xmax = float(max(vals_sorted.max() * 1.25, 0.8))
    ax.set_xlim(0.0, xmax)
    ax.text(0.5 + 0.02, len(vals_sorted) - 0.5, "0.5", color=PRIMARY_RED, fontsize=8)

    _apply_axes_style(ax)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals_sorted], fontsize=7.5, padding=3, color=DARK_GRAY)

    fig.text(
        0.5,
        0.01,
        f"Features with SHAP=0 omitted (n={omitted} features)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=DARK_GRAY,
    )

    return _save_png_pdf(fig, output_dir, "fig3_shap_importance")


def generate_fig4_generalisation(
    generalisation_df: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """Generate cross-format generalisation chart using CSV values."""
    if generalisation_df is None or generalisation_df.empty:
        raise ValueError("generalisation_df is empty.")

    fmt_col = _pick_col(generalisation_df, ["format", "Format"])
    model_col = _pick_col(generalisation_df, ["model", "Model"])
    recall_col = _pick_col(generalisation_df, ["recall", "Recall"])
    auc_col = _pick_col(generalisation_df, ["roc_auc", "roc-auc", "ROC_AUC", "auc"])
    if None in {fmt_col, model_col, recall_col, auc_col}:
        raise ValueError("generalisation_results.csv missing required columns.")

    df = generalisation_df.copy()
    df = df[df[model_col].astype(str).str.contains("XGBoost", case=False, na=False)]

    fmt_order = ["ZIP", "APK", "RAR", "7z"]
    rows = []
    for fmt in fmt_order:
        match = df[df[fmt_col].astype(str).str.strip().str.upper() == fmt.upper()]
        if len(match) == 0:
            raise ValueError(f"Missing XGBoost row for format: {fmt}")
        rows.append(match.iloc[0])

    chart_df = pd.DataFrame(rows)
    chart_df[recall_col] = chart_df[recall_col].astype(float)
    chart_df[auc_col] = chart_df[auc_col].astype(float)

    xlabels = ["ZIP", "APK", "RAR*", "7z*"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.5),
        sharey=True,
        constrained_layout=True,
    )

    for ax, metric_col, title in [
        (axes[0], recall_col, "Recall by Archive Format"),
        (axes[1], auc_col, "ROC-AUC by Archive Format"),
    ]:
        vals = chart_df[metric_col].astype(float).to_numpy()
        colors = []
        for val in vals:
            if val >= 0.99:
                colors.append(SUCCESS_GREEN)
            elif val >= 0.70:
                colors.append(PRIMARY_BLUE)
            elif val >= 0.50:
                colors.append(AMBER)
            else:
                colors.append(PRIMARY_RED)

        bars = ax.bar(xlabels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0.0, 1.15)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xlabel("Archive Format", fontsize=9)
        ax.set_title(title, fontsize=10)
        _apply_axes_style(ax)
        ax.bar_label(bars, fmt="%.4f", fontsize=7.5, padding=3, color=DARK_GRAY)

    fig.text(
        0.5,
        0.01,
        "* Low recall due to threshold miscalibration under distribution shift; "
        "AUC confirms signal transfer",
        ha="center",
        va="bottom",
        fontsize=8,
        color=DARK_GRAY,
    )

    return _save_png_pdf(fig, output_dir, "fig4_generalisation_chart")


def generate_table1_baseline(
    baseline_df: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """Generate baseline-vs-ML comparison table figure from CSV."""
    if baseline_df is None or baseline_df.empty:
        raise ValueError("baseline_df is empty.")

    cols_present = [
        c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "TP", "FP", "TN", "FN"]
        if c in baseline_df.columns
    ]
    if not cols_present:
        raise ValueError("No expected baseline columns found.")

    display_df = baseline_df[cols_present].copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if c in display_df.columns:
            display_df[c] = display_df[c].map(lambda x: _format_metric(x, digits=4))

    n_data_rows = len(display_df)
    n_rows_including_header = n_data_rows + 1
    fig_height = n_rows_including_header * 0.13 + 0.20 + 0.10

    fig, ax = plt.subplots(figsize=(7.0, fig_height), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MED_GRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.15)
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)
            cell.set_height(0.13)

    return _save_png_pdf(fig, output_dir, "table1_baseline_comparison")


def generate_table2_dataset(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """Generate dataset statistics table using directory and label counts."""
    if labels_df is None or labels_df.empty:
        raise ValueError("labels_df is empty.")

    synthetic_mal = _count_files("data/raw/malicious")
    synthetic_ben = _count_files("data/raw/benign")
    real_world = _count_files("data/real_world_validation")

    malicious_count = int((labels_df["label"] == 1).sum())
    benign_count = int((labels_df["label"] == 0).sum())
    total_count = int(len(labels_df))

    def use_or_fallback(dir_count: int | None, fallback_val: int) -> int:
        return dir_count if dir_count is not None else fallback_val

    rows = [
        ["Synthetic malicious files", use_or_fallback(synthetic_mal, malicious_count), "data/raw/malicious or label fallback"],
        ["Synthetic benign files", use_or_fallback(synthetic_ben, benign_count), "data/raw/benign or label fallback"],
        ["Real-world validation files", use_or_fallback(real_world, total_count), "data/real_world_validation or label fallback"],
        ["Processed malicious labels", malicious_count, "labels_df where label=1"],
        ["Processed benign labels", benign_count, "labels_df where label=0"],
        ["Total processed samples", total_count, "len(labels_df)"],
    ]

    cols = ["Component", "Count", "Derivation"]

    n_rows = 7
    fig_height = n_rows * 0.13 + 0.20 + 0.30

    fig, ax = plt.subplots(figsize=(7.0, fig_height), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        colWidths=[0.45, 0.15, 0.40],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MED_GRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.15)
        else:
            cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)
            cell.set_height(0.13)

    return _save_png_pdf(fig, output_dir, "table2_dataset_statistics")


def generate_table3_prevalence(
    realworld_df: pd.DataFrame | None,
    output_dir: str,
) -> tuple[str, str]:
    """Generate prevalence breakdown table using real-world labels if present."""
    used_fallback = False

    if realworld_df is not None and not realworld_df.empty:
        if "label" not in realworld_df.columns:
            raise ValueError("realworld_df missing label column.")

        signal_col = _pick_col(realworld_df, ["signal", "Signal"])
        total = int(len(realworld_df))
        detected = int((realworld_df["label"] == 1).sum())

        if signal_col is not None:
            positives = realworld_df[realworld_df["label"] == 1]
            by_signal = positives.groupby(signal_col).size().to_dict()
        else:
            by_signal = {}

        goot_count = int(by_signal.get("gootloader", 0))
        entropy_count = int(by_signal.get("entropy", 0))
        unknown_count = int(by_signal.get("unknown_method", 0))
        mismatch_count = int(by_signal.get("mismatch", 0))
    else:
        used_fallback = True
        total = PREVALENCE_TOTAL
        detected = PREVALENCE_DETECTED
        goot_count = PREVALENCE_GOOTLOADER
        entropy_count = PREVALENCE_ENTROPY
        unknown_count = PREVALENCE_UNKNOWN_METHOD
        mismatch_count = PREVALENCE_MISMATCH
        print(
            "WARNING: Using hardcoded prevalence values - run "
            "verify_realworld.py to regenerate from data"
        )

    non_evasion = int(total - detected)

    def pct(val: int, denom: int) -> str:
        if denom <= 0:
            return "0.0%"
        return f"{(100.0 * val / denom):.1f}%"

    rows = [
        ["Gootloader EOCD chaining (EOCD > 1)", goot_count, pct(goot_count, total), "eocd_count"],
        ["High entropy anomaly", entropy_count, pct(entropy_count, total), "data_entropy_shannon"],
        ["Undefined LFH method code", unknown_count, pct(unknown_count, total), "lf_unknown_method"],
        ["LFH/CDH mismatch", mismatch_count, pct(mismatch_count, total), "method_mismatch"],
        ["Total detected evasion", detected, pct(detected, total), "label==1"],
        ["Non-evasion / out-of-scope", non_evasion, pct(non_evasion, total), "label==0"],
    ]

    cols = ["Signal Type", "Count", "Share", "Feature"]

    n_data_rows = len(rows)
    fig_height = (n_data_rows + 1) * 0.13 + 0.20 + 0.10

    fig, ax = plt.subplots(figsize=(7.0, fig_height), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        colWidths=[0.48, 0.10, 0.15, 0.27],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(MED_GRAY)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(PRIMARY_BLUE)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.15)
        else:
            if row == 5:
                cell.set_facecolor(LIGHT_GREEN_BG)
            elif row == 6:
                cell.set_facecolor(LIGHT_AMBER_BG)
            else:
                cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)
            cell.set_height(0.13)

    if used_fallback:
        fig.text(
            0.5,
            0.01,
            "Fallback prevalence constants used due to missing realworld_labels.csv",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PRIMARY_RED,
        )

    return _save_png_pdf(fig, output_dir, "table3_prevalence_breakdown")


def generate_table4_generalisation(
    generalisation_df: pd.DataFrame,
    output_dir: str,
) -> tuple[str, str]:
    """Generate generalisation results table from CSV rows."""
    if generalisation_df is None or generalisation_df.empty:
        raise ValueError("generalisation_df is empty.")

    fmt_col = _pick_col(generalisation_df, ["format", "Format"])
    model_col = _pick_col(generalisation_df, ["model", "Model"])
    recall_col = _pick_col(generalisation_df, ["recall", "Recall"])
    precision_col = _pick_col(generalisation_df, ["precision", "Precision"])
    f1_col = _pick_col(generalisation_df, ["f1", "F1"])
    auc_col = _pick_col(generalisation_df, ["roc_auc", "roc-auc", "ROC_AUC", "auc"])
    if None in {fmt_col, model_col, recall_col, precision_col, f1_col, auc_col}:
        raise ValueError("generalisation_results.csv missing required columns.")

    rows = []
    for _, rec in generalisation_df.iterrows():
        fmt = str(rec[fmt_col])
        model = str(rec[model_col])
        recall = _safe_float(rec[recall_col])
        precision = _safe_float(rec[precision_col])
        f1 = _safe_float(rec[f1_col])
        auc = _safe_float(rec[auc_col])

        if model == "Transformer" and not np.isnan(auc) and auc < 0.5:
            note = "Flagged all samples as malicious"
        elif model == "XGBoost" and not np.isnan(recall) and recall == 1.0:
            note = "ZIP-based, full feature transfer"
        elif (
            not np.isnan(auc)
            and auc > 0.95
            and not np.isnan(recall)
            and recall < 0.5
        ):
            note = "Signal present, threshold miscalibration"
        else:
            note = ""

        rows.append([
            fmt,
            model,
            _format_metric(recall, 4),
            _format_metric(precision, 4),
            _format_metric(f1, 4),
            _format_metric(auc, 4),
            note,
        ])

    cols = ["Format", "Model", "Recall", "Precision", "F1", "ROC-AUC", "Notes"]

    n_data_rows = len(rows)
    fig_height = (n_data_rows + 1) * 0.13 + 0.20 + 0.10

    fig, ax = plt.subplots(figsize=(7.0, fig_height), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        colWidths=[0.12, 0.14, 0.09, 0.11, 0.08, 0.10, 0.36],
        cellLoc="center",
        loc="center",
    )
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

        cell.set_height(0.13)
        cell.set_facecolor("white" if row % 2 else LIGHT_GRAY)

        if col == 5:
            val = _safe_float(rows[row - 1][5])
            if np.isnan(val):
                continue
            if val >= 0.95:
                cell.set_text_props(color=SUCCESS_GREEN, fontweight="bold")
            elif val < 0.50:
                cell.set_text_props(color=PRIMARY_RED, fontweight="bold")
            else:
                cell.set_text_props(color=PRIMARY_BLUE)

    return _save_png_pdf(fig, output_dir, "table4_generalisation_results")


def verify_outputs(output_dir: str) -> tuple[int, int, list[str]]:
    """Verify output resolution and PDF pairing for all PNG figures."""
    out_dir = Path(output_dir)
    png_files = sorted(out_dir.glob("*.png"))
    warnings: list[str] = []
    pdf_ok_count = 0

    print("\nVerification report:")
    for png in png_files:
        img = plt.imread(png)
        width_px, height_px = int(img.shape[1]), int(img.shape[0])

        if "fig" in png.name or "table" in png.name:
            if width_px < 2000:
                warning = (
                    f"WARNING: {png.name} may be low resolution "
                    f"({width_px}px wide)"
                )
                print(warning)
                warnings.append(warning)
            else:
                print(f"OK: {png.name} {width_px}x{height_px}px")

        pdf_path = png.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"PDF OK: {pdf_path}")
            pdf_ok_count += 1
        else:
            warning = f"MISSING PDF: {pdf_path}"
            print(warning)
            warnings.append(warning)

    return len(png_files), pdf_ok_count, warnings


def _run_step(step_name: str, fn: Any, *args: Any) -> bool:
    """Run one output-generation step with isolated failure handling."""
    try:
        png_path, pdf_path = fn(*args)
        print(f"OK: {step_name} saved -> {png_path} and {pdf_path}")
        return True
    except Exception as exc:
        print(f"FAILED: {step_name} - {exc}")
        traceback.print_exc()
        return False


def main() -> None:
    """Entry point for complete figure regeneration workflow."""
    configure_style()
    output_dir = "paper/figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ZombieGuard - Generating all paper figures/tables")
    print("=" * 60)

    # STEP 1 - Load all data sources at startup.
    model = None
    features_df = None
    labels_df = None
    merged = None
    baseline_df = None
    generalisation_df = None
    realworld_df = None
    shap_values = None
    mean_shap = None
    feature_names: list[str] = []

    model_path = "models/xgboost_model.pkl"
    if Path(model_path).exists():
        try:
            model = joblib.load(model_path)
            print(f"Loaded model: {model_path}")
        except Exception as exc:
            print(f"ERROR: Could not load model at {model_path}: {exc}")
    else:
        print(f"ERROR: Missing model file: {model_path}")

    features_df = _read_csv_or_none("data/processed/features.csv", "features_df")
    labels_df = _read_csv_or_none("data/processed/labels.csv", "labels_df")
    baseline_df = _read_csv_or_none(
        "paper/figures/table1_baseline_comparison.csv",
        "baseline_df",
    )
    generalisation_df = _read_csv_or_none(
        "paper/figures/generalisation_results.csv",
        "generalisation_df",
    )

    realworld_path = "data/realworld_labels.csv"
    if os.path.exists(realworld_path):
        realworld_df = _read_csv_or_none(realworld_path, "realworld_df")
    else:
        print("INFO: data/realworld_labels.csv not found. Table 3 fallback may be used.")

    if features_df is not None and labels_df is not None:
        try:
            merged = features_df.merge(labels_df, on="filename")
            print(f"Merged features/labels rows: {len(merged)}")
        except Exception as exc:
            print(f"ERROR: Failed to merge features and labels: {exc}")

    if model is not None and merged is not None:
        missing_cols = [col for col in FEATURE_COLS if col not in merged.columns]
        if missing_cols:
            print(
                "ERROR: Missing columns for SHAP computation in merged dataframe: "
                f"{missing_cols}"
            )
        else:
            x = merged[FEATURE_COLS].copy()
            for col in [
                "method_mismatch",
                "declared_vs_entropy_flag",
                "lf_unknown_method",
                "any_crc_mismatch",
                "is_encrypted",
            ]:
                if col in x.columns:
                    x[col] = x[col].astype(float)

            feature_names = [FEATURE_LABELS.get(col, col) for col in FEATURE_COLS]
            x.columns = feature_names

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x)
                mean_shap = np.abs(shap_values).mean(axis=0)
                print("Computed SHAP values from model live.")
            except Exception as exc:
                print(f"ERROR: Failed to compute SHAP values live: {exc}")

    expected_variant_count = _extract_variant_count("data/generate_zombie_samples.py")

    # Generation steps with isolated failures.
    successful_steps = 0
    total_steps = 7

    if _run_step("Figure 1", generate_fig1_zip_header, output_dir):
        successful_steps += 1

    if _run_step("Figure 2", generate_fig2_taxonomy, output_dir, expected_variant_count):
        successful_steps += 1

    if shap_values is not None and mean_shap is not None and feature_names:
        if _run_step(
            "Figure 3",
            generate_fig3_shap,
            shap_values,
            feature_names,
            mean_shap,
            output_dir,
        ):
            successful_steps += 1
    else:
        print("FAILED: Figure 3 - SHAP inputs unavailable; skipping.")

    if generalisation_df is not None:
        if _run_step(
            "Figure 4",
            generate_fig4_generalisation,
            generalisation_df,
            output_dir,
        ):
            successful_steps += 1
    else:
        print("FAILED: Figure 4 - generalisation_df unavailable; skipping.")

    if baseline_df is not None:
        if _run_step(
            "Table 1",
            generate_table1_baseline,
            baseline_df,
            output_dir,
        ):
            successful_steps += 1
    else:
        print("FAILED: Table 1 - baseline_df unavailable; skipping.")

    if features_df is not None and labels_df is not None:
        if _run_step(
            "Table 2",
            generate_table2_dataset,
            features_df,
            labels_df,
            output_dir,
        ):
            successful_steps += 1
    else:
        print("FAILED: Table 2 - features/labels unavailable; skipping.")

    if _run_step("Table 3", generate_table3_prevalence, realworld_df, output_dir):
        successful_steps += 1

    if generalisation_df is not None:
        if _run_step(
            "Table 4",
            generate_table4_generalisation,
            generalisation_df,
            output_dir,
        ):
            successful_steps += 1
    else:
        print("FAILED: Table 4 - generalisation_df unavailable; skipping.")

    png_count, pdf_count, warnings = verify_outputs(output_dir)

    print("\n" + "=" * 60)
    print("Final summary")
    print("=" * 60)
    print(f"Successful steps: {successful_steps}/{total_steps + 1}")
    print(f"Total figures generated (PNG found): {png_count}")
    print(f"Total PDFs generated (paired with PNG): {pdf_count}")

    if warnings:
        print("Warnings:")
        for warn in warnings:
            print(f"  - {warn}")
        ready = "No"
    else:
        print("Warnings: None")
        ready = "Yes"

    print(f"READY FOR SUBMISSION: {ready}")


if __name__ == "__main__":
    main()
