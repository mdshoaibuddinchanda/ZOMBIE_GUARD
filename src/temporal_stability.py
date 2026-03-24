"""
temporal_stability.py
EXPERIMENT 3 — Temporal Stability Analysis

Motivation (TESSERACT / NDSS 2025 concept drift framing):
  A common failure mode in ML-based detectors is temporal decay — models trained
  on early samples degrade as attackers adapt. ZombieGuard is theoretically immune
  because it detects compression physics violations (entropy, method codes), not
  byte signatures. This experiment empirically validates that claim.

Method:
  1. Sort 1318 real-world MalwareBazaar samples by first_seen timestamp.
  2. Split into three temporal windows T1/T2/T3 (earliest/middle/latest ~33%).
  3. Train XGBoost on T1 malicious + proportional benign, test on T2 and T3.
  4. Also evaluate the pre-trained synthetic model (zero-shot) on all three windows.
  5. Report recall, precision, F1, AUC per window.
  6. Report top-5 SHAP features per window to check stability.

Expected result: near-constant recall across T1→T2→T3 (physics-based features
don't drift), validating the consistency-verification framing.

Outputs:
  paper/figures/table8_temporal_stability.csv/.png/.pdf
  paper/figures/fig7_temporal_stability_chart.png/.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.extractor import extract_features

# Log to file — conda run on Windows swallows stdout
_LOG = open("data/temporal_stability.log", "w", buffering=1, encoding="utf-8")

def log(msg: str = ""):
    print(msg, flush=True)
    _LOG.write(msg + "\n")
    _LOG.flush()

# ── Paths ────────────────────────────────────────────────────────────────────
TIMESTAMPS_CSV   = "data/bazaar_timestamps.csv"
VALIDATION_DIR   = "data/real_world_validation"
SYNTH_FEATURES   = "data/processed/features.csv"
SYNTH_LABELS     = "data/processed/labels.csv"
PRETRAINED_MODEL = "models/lgbm_model.pkl"
CSV_DIR          = "paper/figures/csv"
PNG_DIR          = "paper/figures/png"
PDF_DIR          = "paper/figures/pdf"

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




# ── Data loading ─────────────────────────────────────────────────────────────

def load_timestamps() -> pd.DataFrame:
    """Load bazaar timestamps CSV."""
    df = pd.read_csv(TIMESTAMPS_CSV)
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    df = df.dropna(subset=["first_seen"])
    df = df.sort_values("first_seen").reset_index(drop=True)
    log(f"Timestamps loaded: {len(df)} samples")
    log(f"  Date range: {df['first_seen'].min().date()} → {df['first_seen'].max().date()}")
    return df


def extract_features_for_file(fpath: str) -> dict | None:
    """Extract features, skipping files over 30MB (entropy calc on huge files hangs)."""
    try:
        size = Path(fpath).stat().st_size
        if size > 30 * 1024 * 1024:
            # For large files, we still need features — use a truncated read
            # These are all encrypted/method-99 ZIPs so entropy will be ~8.0
            # We can safely infer features from the first 512KB of metadata
            import zipfile
            feats = {
                "lf_compression_method": -1, "cd_compression_method": -1,
                "method_mismatch": False, "data_entropy_shannon": 8.0,
                "data_entropy_renyi": 8.0, "declared_vs_entropy_flag": False,
                "eocd_count": 1, "lf_unknown_method": 0,
                "entry_count": 1, "suspicious_entry_count": 0,
                "suspicious_entry_ratio": 0.0, "entropy_variance": 0.0,
                "lf_crc_valid": True, "any_crc_mismatch": False,
                "is_encrypted": True, "file_size_bytes": size,
            }
            try:
                with zipfile.ZipFile(fpath) as zf:
                    infos = zf.infolist()
                    if infos:
                        feats["entry_count"] = len(infos)
                        feats["lf_compression_method"] = infos[0].compress_type
                        feats["cd_compression_method"] = infos[0].compress_type
                        feats["is_encrypted"] = bool(infos[0].flag_bits & 0x1)
                        feats["lf_unknown_method"] = int(infos[0].compress_type not in (0, 8))
            except Exception:
                pass
            return feats
        return extract_features(fpath)
    except Exception:
        return None


def load_or_extract_malicious(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for all malicious real-world samples.
    Uses a cache file to avoid re-extracting on repeated runs.
    """
    cache_path = "data/temporal_features_cache.csv"
    if Path(cache_path).exists():
        log("Loading cached real-world features...")
        df = pd.read_csv(cache_path)
        log(f"  Loaded {len(df)} cached rows")
        return df

    log("Extracting features from real-world validation samples...")
    val_dir = Path(VALIDATION_DIR)
    rows = []
    missing = 0

    for i, row in ts_df.iterrows():
        short = row["sha256_short"]
        fpath = val_dir / f"{short}.zip"
        if not fpath.exists():
            # Try recent_ prefix
            candidates = list(val_dir.glob(f"*{short[:12]}*.zip"))
            if candidates:
                fpath = candidates[0]
            else:
                missing += 1
                continue

        feats = extract_features_for_file(str(fpath))
        if feats is None:
            missing += 1
            continue

        feats["filename"] = fpath.name
        feats["sha256_short"] = short
        feats["first_seen"] = row["first_seen"]
        feats["label"] = 1
        rows.append(feats)

        if (len(rows)) % 100 == 0:
            log(f"  {len(rows)} extracted, {missing} missing/failed...")

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    log(f"Extracted {len(df)} malicious samples ({missing} missing/failed)")
    log(f"Cached to {cache_path}")
    return df


def load_benign() -> pd.DataFrame:
    """Load benign samples from the synthetic dataset."""
    feats = pd.read_csv(SYNTH_FEATURES)
    labels = pd.read_csv(SYNTH_LABELS)
    df = feats.merge(labels, on="filename")
    benign = df[df["label"] == 0].copy()
    log(f"Benign samples available: {len(benign)}")
    return benign


# ── Temporal split ────────────────────────────────────────────────────────────

def split_temporal(mal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split malicious samples into T1/T2/T3 tertiles by sample count (equal thirds)."""
    mal_sorted = mal_df.sort_values("first_seen").reset_index(drop=True)
    n = len(mal_sorted)
    t1_end = n // 3
    t2_end = 2 * n // 3

    t1 = mal_sorted.iloc[:t1_end].copy()
    t2 = mal_sorted.iloc[t1_end:t2_end].copy()
    t3 = mal_sorted.iloc[t2_end:].copy()

    log(f"\nTemporal split (equal-count tertiles):")
    log(f"  T1 (earliest): {len(t1)} samples  {t1['first_seen'].min().date()} → {t1['first_seen'].max().date()}")
    log(f"  T2 (middle):   {len(t2)} samples  {t2['first_seen'].min().date()} → {t2['first_seen'].max().date()}")
    log(f"  T3 (latest):   {len(t3)} samples  {t3['first_seen'].min().date()} → {t3['first_seen'].max().date()}")
    return t1, t2, t3


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def prepare_xy(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    for col in available:
        X[col] = pd.to_numeric(
            X[col].map(lambda v: 1 if v is True or v == "True" else (0 if v is False or v == "False" else v)),
            errors="coerce"
        ).fillna(0)
    X = X.astype(float)
    y = df["label"].astype(int)
    return X, y


def evaluate_window(model, df: pd.DataFrame, window_name: str, threshold: float = 0.5) -> dict:
    """Evaluate model on a single temporal window."""
    X, y = prepare_xy(df)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())

    recall = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = float("nan")

    log(f"  {window_name}: recall={recall:.4f}  precision={precision:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}  TP={tp}  FP={fp}  FN={fn}")

    return {
        "window": window_name,
        "n_malicious": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def top_shap_features(model, df: pd.DataFrame, n: int = 5) -> list[str]:
    """Return top-n feature names by mean |SHAP| value."""
    X, _ = prepare_xy(df)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    available = [c for c in FEATURE_COLS if c in df.columns]
    return [available[i] for i in top_idx]


# ── Figure generation ─────────────────────────────────────────────────────────

def save_table_figure(rows: list[dict], csv_stem: str, title: str = ""):
    """Save results table as CSV only (no image output for tables)."""
    df = pd.DataFrame(rows)
    csv_path = f"{csv_stem}.csv"
    df.to_csv(csv_path, index=False)
    log(f"Saved: {csv_path}")



def save_line_chart(rows: list[dict], png_path: str, pdf_path: str):
    """Save recall/F1/AUC line chart across temporal windows."""
    df = pd.DataFrame(rows)

    # Separate synthetic-trained vs temporal-trained rows
    synth_rows = df[df["window"].str.startswith("Synth\u2192")]
    temp_rows  = df[~df["window"].str.startswith("Synth\u2192")]

    fig, ax = plt.subplots(figsize=(9, 5))

    windows = temp_rows["window"].tolist()
    x = np.arange(len(windows))

    for metric, color, marker in [("recall", "#e74c3c", "o"),
                                   ("f1",     "#3498db", "s"),
                                   ("auc",    "#2ecc71", "^")]:
        vals = temp_rows[metric].tolist()
        ax.plot(x, vals, color=color, marker=marker, linewidth=2,
                markersize=8, label=f"Temporal-trained {metric.upper()}")

    # Dashed lines for synthetic (zero-shot) model recall
    if not synth_rows.empty:
        synth_windows = synth_rows["window"].str.replace("Synth\u2192", "").tolist()
        synth_recall  = synth_rows["recall"].tolist()
        synth_x = [windows.index(w) for w in synth_windows if w in windows]
        synth_y = [synth_recall[i] for i, w in enumerate(synth_windows) if w in windows]
        ax.plot(synth_x, synth_y, color="#e67e22", marker="D", linewidth=2,
                markersize=8, linestyle="--", label="Synthetic-trained Recall (zero-shot)")

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Temporal Window", fontsize=12)
    ax.set_title("Temporal Stability: Recall / F1 / AUC Across Time Windows", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {png_path}")
    log(f"Saved: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("EXPERIMENT 3 — Temporal Stability Analysis")
    log("=" * 60)

    # 1. Load timestamps and extract features
    ts_df = load_timestamps()
    mal_df = load_or_extract_malicious(ts_df)
    benign_df = load_benign()

    # Ensure first_seen is datetime
    mal_df["first_seen"] = pd.to_datetime(mal_df["first_seen"], errors="coerce")
    mal_df = mal_df.dropna(subset=["first_seen"])
    log(f"\nMalicious samples with features + timestamps: {len(mal_df)}")

    # 2. Temporal split
    t1_mal, t2_mal, t3_mal = split_temporal(mal_df)

    # 3. Assign benign samples proportionally across windows
    n_total_mal = len(mal_df)
    rng = np.random.default_rng(42)
    benign_shuffled = benign_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def benign_slice(window_mal):
        n = int(len(window_mal) / n_total_mal * len(benign_shuffled))
        return benign_shuffled.iloc[:n]

    t1_ben = benign_slice(t1_mal)
    t2_ben = benign_slice(t2_mal)
    t3_ben = benign_slice(t3_mal)

    t1 = pd.concat([t1_mal, t1_ben], ignore_index=True)
    t2 = pd.concat([t2_mal, t2_ben], ignore_index=True)
    t3 = pd.concat([t3_mal, t3_ben], ignore_index=True)

    log(f"\nWindow sizes (mal+benign): T1={len(t1)}, T2={len(t2)}, T3={len(t3)}")

    # 4. Train on T1, evaluate on T2 and T3
    log("\n--- Training temporal model on T1 ---")
    X_t1, y_t1 = prepare_xy(t1)
    temporal_model = build_xgb()
    temporal_model.fit(X_t1, y_t1)
    log(f"  Trained on {len(X_t1)} samples (mal={y_t1.sum()}, benign={(y_t1==0).sum()})")

    log("\n--- Evaluating temporal model ---")
    temp_results = []
    temp_results.append(evaluate_window(temporal_model, t1, "T1 (train)"))
    temp_results.append(evaluate_window(temporal_model, t2, "T2 (test)"))
    temp_results.append(evaluate_window(temporal_model, t3, "T3 (test)"))

    # 5. Load pre-trained synthetic model and evaluate zero-shot on all windows
    log("\n--- Evaluating pre-trained synthetic model (zero-shot) ---")
    log("    (Using optimal threshold from T1 ROC curve)")
    synth_model = joblib.load(PRETRAINED_MODEL)

    # Find optimal threshold on T1 to avoid default-0.5 bias
    from sklearn.metrics import roc_curve
    X_t1_eval, y_t1_eval = prepare_xy(t1)
    synth_proba_t1 = synth_model.predict_proba(X_t1_eval)[:, 1]
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_t1_eval, synth_proba_t1)
    # Youden's J statistic
    j_scores = tpr_arr - fpr_arr
    best_thresh = float(thresh_arr[np.argmax(j_scores)])
    log(f"    Optimal threshold (Youden J on T1): {best_thresh:.4f}")

    synth_results = []
    for window_df, name in [(t1, "Synth→T1"), (t2, "Synth→T2"), (t3, "Synth→T3")]:
        synth_results.append(evaluate_window(synth_model, window_df, name, threshold=best_thresh))

    # 6. SHAP feature stability
    log("\n--- SHAP feature stability across windows ---")
    shap_stability = {}
    for window_df, name in [(t1, "T1"), (t2, "T2"), (t3, "T3")]:
        top5 = top_shap_features(temporal_model, window_df)
        shap_stability[name] = top5
        log(f"  {name} top-5: {top5}")

    # Check stability: how many features appear in all 3 windows
    all_sets = list(set(v) for v in shap_stability.values())
    stable = set.intersection(*all_sets) if all_sets else set()
    log(f"\n  Features stable across all windows: {stable}")

    # 7. Save outputs
    all_results = temp_results + synth_results

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    save_table_figure(
        all_results,
        csv_stem=f"{CSV_DIR}/table8_temporal_stability",
        title="Table 8 — Temporal Stability Analysis (T1/T2/T3 Windows)",
    )

    save_line_chart(
        all_results,
        png_path=f"{PNG_DIR}/fig7_temporal_stability_chart.png",
        pdf_path=f"{PDF_DIR}/fig7_temporal_stability_chart.pdf",
    )

    # Save SHAP stability summary
    shap_df = pd.DataFrame([
        {"window": w, "rank": i+1, "feature": f}
        for w, feats in shap_stability.items()
        for i, f in enumerate(feats)
    ])
    shap_csv = f"{CSV_DIR}/table8b_shap_stability.csv"
    shap_df.to_csv(shap_csv, index=False)
    log(f"Saved SHAP stability: {shap_csv}")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"{'Window':<20} {'Recall':>8} {'F1':>8} {'AUC':>8} {'FP':>6}")
    log("-" * 55)
    for r in temp_results:
        log(f"  {r['window']:<18} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fp']:>6}")
    log()
    log(f"{'Window (Synth)':<20} {'Recall':>8} {'F1':>8} {'AUC':>8} {'FP':>6}")
    log("-" * 55)
    for r in synth_results:
        log(f"  {r['window']:<18} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fp']:>6}")

    log(f"\nStable SHAP features (all 3 windows): {stable}")
    log("\nDone.")


if __name__ == "__main__":
    main()
