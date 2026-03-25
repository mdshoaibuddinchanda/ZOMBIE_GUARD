"""Validate synthetic vs real feature alignment and generate projection plots."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import FEATURE_COLS
from src.realworld_features import load_realworld_features

SYNTH_FEATURES = Path("data/processed/features.csv")
SYNTH_LABELS = Path("data/processed/labels.csv")
OUT_CSV = Path("paper/figures/csv/table_synthetic_real_feature_alignment.csv")
OUT_PNG = Path("paper/figures/png/fig_synthetic_real_feature_space_pca.png")
OUT_PDF = Path("paper/figures/pdf/fig_synthetic_real_feature_space_pca.pdf")


def load_synthetic_dataset() -> pd.DataFrame:
    """Load canonical synthetic feature matrix with labels."""
    synth_features = pd.read_csv(SYNTH_FEATURES)
    synth_labels = pd.read_csv(SYNTH_LABELS)
    synth_df = synth_features.merge(synth_labels, on="filename", how="inner")
    for col in FEATURE_COLS:
        if col not in synth_df.columns:
            synth_df[col] = 0
    return synth_df


def build_alignment_table(
    synth_mal: pd.DataFrame,
    real_mal: pd.DataFrame,
) -> pd.DataFrame:
    """Compute KS, mean, and std comparisons for all model features."""
    rows: list[dict[str, float | str]] = []
    for feature in FEATURE_COLS:
        syn_vals = synth_mal[feature].astype(float).to_numpy()
        real_vals = real_mal[feature].astype(float).to_numpy()

        ks_result: Any = ks_2samp(syn_vals, real_vals, method="auto")
        ks_stat = float(ks_result[0])
        p_value = float(ks_result[1])
        rows.append(
            {
                "feature": feature,
                "synthetic_mal_mean": float(np.mean(syn_vals)),
                "synthetic_mal_std": float(np.std(syn_vals)),
                "real_mal_mean": float(np.mean(real_vals)),
                "real_mal_std": float(np.std(real_vals)),
                "ks_statistic": ks_stat,
                "ks_p_value": p_value,
                "mean_abs_diff": float(abs(np.mean(syn_vals) - np.mean(real_vals))),
            }
        )

    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def create_projection_plot(
    synth_mal: pd.DataFrame,
    real_mal: pd.DataFrame,
    real_ben: pd.DataFrame,
) -> None:
    """Create a PCA projection for synthetic-malicious, real-malicious, real-benign."""
    stacked = pd.concat(
        [
            synth_mal.assign(group="Synthetic malicious"),
            real_mal.assign(group="Real malicious"),
            real_ben.assign(group="Real benign"),
        ],
        ignore_index=True,
    )

    x = stacked[FEATURE_COLS].astype(float)
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    plot_df = pd.DataFrame({
        "pc1": x_pca[:, 0],
        "pc2": x_pca[:, 1],
        "group": stacked["group"],
    })

    plt.figure(figsize=(8.5, 6.5))
    palette = {
        "Synthetic malicious": "#1f77b4",
        "Real malicious": "#d62728",
        "Real benign": "#2ca02c",
    }
    for group, group_df in plot_df.groupby("group"):
        plt.scatter(
            group_df["pc1"],
            group_df["pc2"],
            s=14,
            alpha=0.45,
            c=palette[str(group)],
            label=group,
            edgecolors="none",
        )

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
    plt.title("Synthetic vs Real Feature Space Alignment")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=600)
    plt.savefig(OUT_PDF)
    plt.close()


def main() -> None:
    """Run feature-distribution validation and export reviewer-ready artifacts."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("Running synthetic-real feature distribution validation...")
    logger.info("Loading synthetic dataset...")
    synth_df = load_synthetic_dataset()
    logger.info(f"  ✓ Loaded {len(synth_df)} synthetic samples.")
    
    logger.info("Loading real-world dataset (may take time on first run)...")
    real_df = load_realworld_features(refresh=False)
    logger.info(f"  ✓ Loaded {len(real_df)} real-world samples.")

    synth_mal = synth_df[synth_df["label"] == 1].copy()
    real_mal = real_df[real_df["label"] == 1].copy()
    real_ben = real_df[real_df["label"] == 0].copy()

    logger.info(f"Comparing {len(synth_mal)} synthetic malicious vs {len(real_mal)} real malicious samples...")
    logger.info("Computing KS statistics and feature alignment metrics...")
    table = build_alignment_table(synth_mal, real_mal)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    logger.info(f"  ✓ Alignment table computed.")

    logger.info("Generating PCA projection plot...")
    create_projection_plot(synth_mal, real_mal, real_ben)
    logger.info(f"  ✓ Projection plot generated.")

    logger.info(f"Saved: {OUT_CSV}")
    logger.info(f"Saved: {OUT_PNG}")
    logger.info(f"Saved: {OUT_PDF}")
    print("Top 5 largest KS differences:")
    print(table[["feature", "ks_statistic", "ks_p_value"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
