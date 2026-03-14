import os
import sys

import joblib
import matplotlib
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_PATH = "data/processed/features.csv"
LABELS_PATH = "data/processed/labels.csv"
MODEL_PATH = "models/xgboost_model.pkl"
FIGURES_DIR = "paper/figures"

FEATURE_COLS = [
    "lf_compression_method",
    "cd_compression_method",
    "method_mismatch",
    "data_entropy_shannon",
    "data_entropy_renyi",
    "declared_vs_entropy_flag",
    "eocd_count",
    # file_size_bytes intentionally excluded -
    # size is not a generalizable evasion signal
    "lf_unknown_method",
]

FEATURE_LABELS = {
    "lf_compression_method": "LFH compression method",
    "cd_compression_method": "CDH compression method",
    "method_mismatch": "Method mismatch (LFH vs CDH)",
    "data_entropy_shannon": "Shannon entropy of payload",
    "data_entropy_renyi": "Renyi entropy of payload",
    "declared_vs_entropy_flag": "Declared-vs-entropy mismatch",
    "eocd_count": "EOCD signature count",
    "lf_unknown_method": "Unknown method code (LFH)",
}


def load_data():
    features_df = pd.read_csv(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    merged = features_df.merge(labels_df, on="filename", how="inner")
    for col in ["method_mismatch", "declared_vs_entropy_flag"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(int)
    x = merged[FEATURE_COLS].copy()
    x.columns = [FEATURE_LABELS[column] for column in FEATURE_COLS]
    y = merged["label"]
    return x, y


def generate_shap_plots(x, model):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    # -- Plot 1: Summary plot (all features ranked by importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        x,
        plot_type="bar",
        show=False,
        color="#2E74B5",
    )
    plt.title("ZombieGuard - Feature Importance (SHAP)", fontsize=14, pad=15)
    plt.xlabel("Mean |SHAP value|", fontsize=11)
    plt.tight_layout()
    summary_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_path}")

    # -- Plot 2: Beeswarm (shows direction of each feature's impact)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        x,
        plot_type="dot",
        show=False,
    )
    plt.title("ZombieGuard - SHAP Beeswarm (Feature Direction)", fontsize=14, pad=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(FIGURES_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # -- Plot 3: Waterfall for one specific Zombie ZIP sample
    # Find the first malicious sample in the dataset
    malicious_indices = [
        i
        for i, fname in enumerate(
            pd.read_csv(LABELS_PATH).merge(pd.read_csv(FEATURES_PATH), on="filename")["label"]
        )
        if fname == 1
    ]
    sample_idx = malicious_indices[0] if malicious_indices else 0

    plt.figure(figsize=(10, 5))
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=x.iloc[sample_idx],
        feature_names=list(x.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title("ZombieGuard - Why this file was flagged (SHAP Waterfall)", fontsize=13, pad=15)
    plt.tight_layout()
    waterfall_path = os.path.join(FIGURES_DIR, "shap_waterfall.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {waterfall_path}")

    return shap_values


def print_feature_ranking(shap_values, x):
    import numpy as np

    mean_abs = pd.Series(data=abs(shap_values).mean(axis=0), index=x.columns).sort_values(
        ascending=False
    )

    print("\n-- Feature Importance Ranking (SHAP) ---------------")
    for rank, (feat, value) in enumerate(mean_abs.items(), start=1):
        print(f"  {rank}. {feat:<40} {value:.6f}")
    print("-----------------------------------------------------")
    return mean_abs


if __name__ == "__main__":
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    x, y = load_data()

    print(f"Running SHAP on {len(x)} samples...")
    shap_values = generate_shap_plots(x, model)
    print_feature_ranking(shap_values, x)

    print("\nStep 08 complete.")
    print(f"All figures saved to: {FIGURES_DIR}/")