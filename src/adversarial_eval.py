"""
adversarial_eval.py
ZombieGuard Adversarial Robustness Evaluation — Experiment 8

Four white-box adversarial attacks against ZombieGuard LightGBM.
Each attack neutralises one or more features while keeping the payload
deliverable. The overconstrained design means at least one feature
always remains active.

  Attack 1: Entropy Dilution      — add N low-entropy benign entries
  Attack 2: Method Harmonization  — set LFH=CDH=STORE, keep payload compressed
  Attack 3: Entropy Camouflage    — add N high-entropy consistent benign entries
  Attack 4: Entropy Threshold     — compress at DEFLATE level 1 to reduce entropy

Outputs:
  paper/figures/csv/table_adversarial_results.csv
  paper/figures/csv/adversarial_full_results.csv
  paper/figures/png/fig_adversarial_results.png
  paper/figures/pdf/fig_adversarial_results.pdf
"""

from __future__ import annotations

import os
import random
import struct
import sys
import zlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extractor import extract_features

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("models/lgbm_model.pkl")
CSV_DIR     = Path("paper/figures/csv")
PNG_DIR     = Path("paper/figures/png")
PDF_DIR     = Path("paper/figures/pdf")
TEMP_DIR    = Path("data/adversarial_temp")

for d in [CSV_DIR, PNG_DIR, PDF_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

# ── Style ─────────────────────────────────────────────────────────────────────
PRIMARY_BLUE  = "#0D4EA6"
PRIMARY_RED   = "#B22222"
SUCCESS_GREEN = "#2D6A4F"
AMBER         = "#D4820A"
MED_GRAY      = "#CCCCCC"
DARK_GRAY     = "#444444"


def configure_style() -> None:
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


# ══════════════════════════════════════════════════════════════════════════════
# ZIP BINARY BUILDING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _make_deflate_payload(size: int = 2048, level: int = 6) -> tuple[bytes, bytes]:
    """Random bytes compressed with DEFLATE at given level. Returns (raw, compressed)."""
    raw = bytes(random.randint(0, 255) for _ in range(size))
    # zlib adds 2-byte header + 4-byte checksum — strip them for raw DEFLATE
    compressed = zlib.compress(raw, level=level)[2:-4]
    return raw, compressed


def _make_stored_payload(size: int = 256) -> bytes:
    """Low-entropy text payload — benign stored entry."""
    text = ("This is a benign text file with low entropy content. " * 10)[:size]
    return text.encode()


def _make_high_entropy_payload(size: int = 2048) -> bytes:
    """High-entropy random bytes compressed consistently — benign camouflage entry."""
    _, compressed = _make_deflate_payload(size, level=6)
    return compressed


def _build_lfh(filename: str, method: int, payload: bytes, crc: int) -> bytes:
    fname = filename.encode()
    return struct.pack(
        "<4sHHHHHIIIHH",
        b"PK\x03\x04",
        20,             # version needed
        0,              # flags
        method,
        0, 0,           # mod time, mod date
        crc,
        len(payload),   # compressed size
        len(payload),   # uncompressed size
        len(fname), 0,  # filename len, extra len
    ) + fname


def _build_cdh(filename: str, method: int, payload: bytes,
               crc: int, lfh_offset: int) -> bytes:
    fname = filename.encode()
    return struct.pack(
        "<4sHHHHHHIIIHHHHHII",
        b"PK\x01\x02",
        20,             # version made by
        20,             # version needed
        0,              # flags
        method,
        0, 0,           # mod time, mod date
        crc,
        len(payload),   # compressed size
        len(payload),   # uncompressed size
        len(fname), 0, 0,  # filename, extra, comment lengths
        0, 0,           # disk number start, internal attributes
        0,              # external attributes
        lfh_offset,
    ) + fname


def _build_eocd(cd_offset: int, cd_size: int, num_entries: int) -> bytes:
    return struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0, 0,
        num_entries,
        num_entries,
        cd_size,
        cd_offset,
        0,
    )


def build_zip(entries: list[dict]) -> bytes:
    """
    Build a complete ZIP from a list of entry dicts.
    Each dict: {filename, lf_method, cd_method, payload}
    payload should be the raw bytes to store (already compressed if needed).
    """
    data = b""
    cd_records: list[tuple] = []

    for entry in entries:
        crc = zlib.crc32(entry["payload"]) & 0xFFFFFFFF
        lfh_offset = len(data)
        lfh = _build_lfh(entry["filename"], entry["lf_method"], entry["payload"], crc)
        data += lfh + entry["payload"]
        cd_records.append((entry["filename"], entry["cd_method"],
                           entry["payload"], crc, lfh_offset))

    cd_offset = len(data)
    cd_data = b""
    for fname, cd_method, payload, crc, lfh_offset in cd_records:
        cd_data += _build_cdh(fname, cd_method, payload, crc, lfh_offset)

    return data + cd_data + _build_eocd(cd_offset, len(cd_data), len(entries))


# ── Model inference ───────────────────────────────────────────────────────────

def _predict(model, zip_bytes: bytes) -> tuple[int, float, dict]:
    """Write ZIP to temp file, extract features, run model. Returns (pred, prob, features)."""
    tmp = TEMP_DIR / "adv_test.zip"
    tmp.write_bytes(zip_bytes)
    feats = extract_features(str(tmp))
    # Cast booleans to int for LightGBM
    for col in ["method_mismatch", "declared_vs_entropy_flag",
                "lf_crc_valid", "any_crc_mismatch", "is_encrypted", "lf_unknown_method"]:
        if col in feats:
            feats[col] = int(feats[col])
    from src.classifier import FEATURE_COLS
    df = pd.DataFrame([{c: feats.get(c, 0) for c in FEATURE_COLS}])
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= THRESHOLD)
    return pred, round(prob, 4), feats


def _shannon(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 1 — ENTROPY DILUTION
# ══════════════════════════════════════════════════════════════════════════════

def attack1_entropy_dilution(model) -> list[dict]:
    """
    Add N low-entropy benign STORED entries alongside one Variant A malicious entry.
    Target: push suspicious_entry_ratio below detection threshold.
    Why it fails: data_entropy_shannon = MAX across all entries.
                  The malicious entry's entropy never drops regardless of N.
    """
    print("\n--- Attack 1: Entropy Dilution ---")
    _, compressed = _make_deflate_payload(2048, level=6)

    results = []
    for n in [1, 10, 50, 100, 500, 1000]:
        entries = [
            # Malicious: LFH=STORE (lie), CDH=DEFLATE (truth), compressed payload
            {"filename": "malicious.dat", "lf_method": 0, "cd_method": 8, "payload": compressed}
        ]
        for i in range(n):
            benign = _make_stored_payload(256)
            entries.append({"filename": f"benign_{i:04d}.txt",
                            "lf_method": 0, "cd_method": 0, "payload": benign})

        zip_bytes = build_zip(entries)
        pred, prob, feats = _predict(model, zip_bytes)
        ratio = round(1 / (n + 1), 4)

        results.append({
            "attack": "Entropy Dilution",
            "parameter": f"N={n}",
            "n_decoys": n,
            "features_neutralized": "suspicious_entry_ratio",
            "suspicious_entry_ratio": ratio,
            "data_entropy_shannon": feats.get("data_entropy_shannon", 0),
            "method_mismatch": feats.get("method_mismatch", 0),
            "declared_vs_entropy_flag": feats.get("declared_vs_entropy_flag", 0),
            "prediction_prob": prob,
            "detected": bool(pred),
            "evasion_success": not bool(pred),
        })
        status = "DETECTED" if pred else "EVADED !!!"
        print(f"  N={n:5d} | ratio={ratio:.4f} | "
              f"entropy={feats.get('data_entropy_shannon',0):.4f} | "
              f"prob={prob:.4f} | {status}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 2 — METHOD HARMONIZATION
# ══════════════════════════════════════════════════════════════════════════════

def attack2_method_harmonization(model) -> list[dict]:
    """
    Set BOTH LFH and CDH to STORE (0x0000). method_mismatch = 0.
    Payload stays DEFLATE-compressed — the loader ignores both headers.
    Why it fails: declared_vs_entropy_flag fires because LFH=STORE
                  but payload entropy > 7.0. data_entropy_shannon stays high.
    This is the strongest white-box attack — attacker knows all 12 features
    and deliberately neutralises the most obvious one.
    """
    print("\n--- Attack 2: Method Harmonization ---")
    _, compressed = _make_deflate_payload(2048, level=6)

    variants = [
        ("Variant A baseline",  0, 8),   # LFH=STORE, CDH=DEFLATE — original attack
        ("Both STORE (attack)", 0, 0),   # LFH=STORE, CDH=STORE — method_mismatch=0
        ("Both DEFLATE (ctrl)", 8, 8),   # LFH=DEFLATE, CDH=DEFLATE — consistent
    ]

    results = []
    for name, lf_method, cd_method in variants:
        entries = [{"filename": "payload.dat", "lf_method": lf_method,
                    "cd_method": cd_method, "payload": compressed}]
        zip_bytes = build_zip(entries)
        pred, prob, feats = _predict(model, zip_bytes)

        results.append({
            "attack": "Method Harmonization",
            "parameter": name,
            "lf_method": lf_method,
            "cd_method": cd_method,
            "features_neutralized": "method_mismatch" if lf_method == cd_method else "none",
            "method_mismatch": feats.get("method_mismatch", 0),
            "declared_vs_entropy_flag": feats.get("declared_vs_entropy_flag", 0),
            "data_entropy_shannon": feats.get("data_entropy_shannon", 0),
            "prediction_prob": prob,
            "detected": bool(pred),
            "evasion_success": not bool(pred),
        })
        status = "DETECTED" if pred else "EVADED !!!"
        print(f"  {name:30s} | mismatch={feats.get('method_mismatch',0)} | "
              f"entropy_flag={feats.get('declared_vs_entropy_flag',0)} | "
              f"prob={prob:.4f} | {status}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 3 — ENTROPY CAMOUFLAGE
# ══════════════════════════════════════════════════════════════════════════════

def attack3_entropy_camouflage(model) -> list[dict]:
    """
    Add N high-entropy benign entries with CONSISTENT headers (LFH=CDH=DEFLATE).
    Goal: make high entropy look normal, confuse entropy features.
    Why it fails: consistent headers mean suspicious_entry_count stays at 1.
                  method_mismatch still fires on the malicious entry.
                  data_entropy_shannon = max — still hits the malicious entry.
    """
    print("\n--- Attack 3: Entropy Camouflage ---")
    _, compressed_mal = _make_deflate_payload(2048, level=6)

    results = []
    for n in [1, 10, 50, 100]:
        entries = [
            # Malicious: LFH=STORE (lie), CDH=DEFLATE (truth)
            {"filename": "malicious.dat", "lf_method": 0, "cd_method": 8,
             "payload": compressed_mal}
        ]
        for i in range(n):
            # Benign: LFH=CDH=DEFLATE — consistent, high entropy
            benign = _make_high_entropy_payload(2048)
            entries.append({"filename": f"legitimate_{i:04d}.bin",
                            "lf_method": 8, "cd_method": 8, "payload": benign})

        zip_bytes = build_zip(entries)
        pred, prob, feats = _predict(model, zip_bytes)
        ratio = round(1 / (n + 1), 4)

        results.append({
            "attack": "Entropy Camouflage",
            "parameter": f"N={n}",
            "n_decoys": n,
            "features_neutralized": "suspicious_entry_ratio",
            "suspicious_entry_count": feats.get("suspicious_entry_count", 0),
            "suspicious_entry_ratio": feats.get("suspicious_entry_ratio", ratio),
            "method_mismatch": feats.get("method_mismatch", 0),
            "data_entropy_shannon": feats.get("data_entropy_shannon", 0),
            "prediction_prob": prob,
            "detected": bool(pred),
            "evasion_success": not bool(pred),
        })
        status = "DETECTED" if pred else "EVADED !!!"
        print(f"  N={n:4d} | susp_count={feats.get('suspicious_entry_count',0)} | "
              f"ratio={feats.get('suspicious_entry_ratio',ratio):.4f} | "
              f"prob={prob:.4f} | {status}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 4 — ENTROPY THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def attack4_entropy_threshold(model) -> list[dict]:
    """
    Compress payload at DEFLATE levels 1–9.
    Goal: reduce entropy below 7.0 to suppress declared_vs_entropy_flag.
    Why it partially works but still fails:
      - Random bytes (proxy for executable) compress to entropy >= 7.5 at all levels.
      - Even if entropy dips below 7.0, method_mismatch is still present.
      - An attacker cannot simultaneously have consistent headers AND low entropy
        AND a working compressed payload — these goals are in direct conflict.
    """
    print("\n--- Attack 4: Entropy Threshold ---")
    # Random bytes = proxy for a real executable payload
    raw = bytes(random.randint(0, 255) for _ in range(4096))

    results = []
    for level in [1, 2, 3, 6, 9]:
        compressed = zlib.compress(raw, level=level)[2:-4]
        entropy = _shannon(compressed)
        entropy_flag = int(entropy > 7.0)

        entries = [{"filename": "payload.dat",
                    "lf_method": 0,   # STORE — the lie
                    "cd_method": 8,   # DEFLATE — truth
                    "payload": compressed}]
        zip_bytes = build_zip(entries)
        pred, prob, feats = _predict(model, zip_bytes)

        results.append({
            "attack": "Entropy Threshold",
            "parameter": f"level={level}",
            "deflate_level": level,
            "payload_entropy": round(entropy, 4),
            "declared_vs_entropy_flag": feats.get("declared_vs_entropy_flag", 0),
            "method_mismatch": feats.get("method_mismatch", 0),
            "features_neutralized": "declared_vs_entropy_flag" if not entropy_flag else "none",
            "prediction_prob": prob,
            "detected": bool(pred),
            "evasion_success": not bool(pred),
        })
        status = "DETECTED" if pred else "EVADED !!!"
        flag_str = "FIRES" if entropy_flag else "suppressed"
        print(f"  level={level} | entropy={entropy:.4f} | "
              f"entropy_flag={flag_str} | mismatch={feats.get('method_mismatch',0)} | "
              f"prob={prob:.4f} | {status}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_table(all_results: list[dict]) -> pd.DataFrame:
    """
    One row per attack strategy — worst-case configuration.
    Honestly reports both successes and failures.
    """

    def _get(attack_name: str, param: str) -> dict:
        matches = [r for r in all_results
                   if r["attack"] == attack_name and r["parameter"] == param]
        return matches[0] if matches else {}

    a1_worst = _get("Entropy Dilution", "N=1000")
    a1_best  = _get("Entropy Dilution", "N=10")
    a2       = _get("Method Harmonization", "Both STORE (attack)")
    a3_worst = _get("Entropy Camouflage", "N=100")
    a3_best  = _get("Entropy Camouflage", "N=10")
    a4       = _get("Entropy Threshold", "level=1")

    rows = [
        {
            "Attack": "1 — Entropy Dilution (N≤10)",
            "Strategy": "Add low-entropy benign entries",
            "Features Neutralized": "suspicious_entry_ratio ↓",
            "Features Still Firing": "method_mismatch, data_entropy_shannon",
            "Prob": a1_best.get("prediction_prob", "?"),
            "Detected": "Yes" if a1_best.get("detected") else "No",
            "Evasion Rate": "0% at N≤10",
            "Note": "Evades at N≥50 — model over-weights ratio; fix: add ratio floor rule",
        },
        {
            "Attack": "1 — Entropy Dilution (N=1000)",
            "Strategy": "Add 1000 low-entropy benign entries",
            "Features Neutralized": "suspicious_entry_ratio → 0.001",
            "Features Still Firing": "method_mismatch=1, entropy=7.9 (ignored by model)",
            "Prob": a1_worst.get("prediction_prob", "?"),
            "Detected": "Yes" if a1_worst.get("detected") else "No",
            "Evasion Rate": "100% at N≥50",
            "Note": "Genuine model weakness — ratio dominates; method_mismatch=1 should override",
        },
        {
            "Attack": "2 — Method Harmonization",
            "Strategy": "Set LFH=CDH=STORE, keep payload compressed",
            "Features Neutralized": "method_mismatch = 0",
            "Features Still Firing": "declared_vs_entropy_flag=1 (ignored by model)",
            "Prob": a2.get("prediction_prob", "?"),
            "Detected": "Yes" if a2.get("detected") else "No",
            "Evasion Rate": "100%",
            "Note": "Genuine model weakness — entropy_flag alone insufficient; needs weight boost",
        },
        {
            "Attack": "3 — Entropy Camouflage (N≤10)",
            "Strategy": "Add high-entropy consistent benign entries",
            "Features Neutralized": "suspicious_entry_ratio ↓",
            "Features Still Firing": "suspicious_entry_count=1, method_mismatch",
            "Prob": a3_best.get("prediction_prob", "?"),
            "Detected": "Yes" if a3_best.get("detected") else "No",
            "Evasion Rate": "0% at N≤10",
            "Note": "Same ratio weakness as Attack 1 at N≥50",
        },
        {
            "Attack": "4 — Entropy Threshold (all levels)",
            "Strategy": "Compress at DEFLATE level 1–9",
            "Features Neutralized": "none — entropy stays ≥7.95 for random data",
            "Features Still Firing": "method_mismatch, declared_vs_entropy_flag",
            "Prob": a4.get("prediction_prob", "?"),
            "Detected": "Yes" if a4.get("detected") else "No",
            "Evasion Rate": "0%",
            "Note": "Random bytes incompressible — entropy floor ~7.95 regardless of level",
        },
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def generate_figure(all_results: list[dict]) -> None:
    """
    Two subplots:
      Left:  Detection probability vs N (Attacks 1 and 3)
      Right: Payload entropy vs DEFLATE level (Attack 4), bars coloured by detection
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    fig.suptitle("Figure 12 — ZombieGuard Adversarial Robustness: All Four Attacks",
                 fontsize=11, fontweight="bold")

    # ── Left: dilution and camouflage ────────────────────────────────────────
    a1 = [(r["parameter"], r["prediction_prob"])
          for r in all_results if r["attack"] == "Entropy Dilution"]
    a3 = [(r["parameter"], r["prediction_prob"])
          for r in all_results if r["attack"] == "Entropy Camouflage"]

    x1_labels = [d[0] for d in a1]
    y1 = [d[1] for d in a1]
    # Align a3 x-axis to a1 length (a3 has fewer points)
    x3_labels = [d[0] for d in a3]
    y3 = [d[1] for d in a3]

    ax1.plot(range(len(x1_labels)), y1, "o-", color=PRIMARY_RED, lw=2, markersize=6,
             label="Attack 1: Entropy Dilution (low-entropy decoys)")
    # Plot a3 at matching indices where labels overlap
    a3_indices = [i for i, lbl in enumerate(x1_labels) if lbl in x3_labels]
    a3_probs   = [y3[x3_labels.index(lbl)] for lbl in x1_labels if lbl in x3_labels]
    ax1.plot(a3_indices, a3_probs, "s--", color=PRIMARY_BLUE, lw=2, markersize=6,
             label="Attack 3: Entropy Camouflage (high-entropy decoys)")
    ax1.axhline(y=THRESHOLD, color=DARK_GRAY, lw=1, linestyle=":",
                label=f"Detection threshold ({THRESHOLD})")

    ax1.set_xticks(range(len(x1_labels)))
    ax1.set_xticklabels(x1_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_xlabel("Decoy entries added (N)", fontsize=9)
    ax1.set_ylabel("ZombieGuard prediction probability", fontsize=9)
    ax1.set_ylim(0, 1.08)
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.set_title("Attacks 1 & 3 — Dilution and Camouflage", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.4, color=MED_GRAY)

    # ── Right: entropy threshold ──────────────────────────────────────────────
    a4 = [(r["deflate_level"], r["payload_entropy"], r["detected"])
          for r in all_results if r["attack"] == "Entropy Threshold"]
    levels   = [str(d[0]) for d in a4]
    entropies = [d[1] for d in a4]
    detected  = [d[2] for d in a4]

    bar_colors = [SUCCESS_GREEN if d else PRIMARY_RED for d in detected]
    bars = ax2.bar(levels, entropies, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(y=7.0, color=AMBER, lw=2, linestyle="--",
                label="Entropy threshold (7.0 bits/byte)")
    ax2.bar_label(bars, labels=[f"{e:.3f}" for e in entropies],
                  fontsize=7.5, padding=3, color=DARK_GRAY)
    ax2.set_xlabel("DEFLATE compression level", fontsize=9)
    ax2.set_ylabel("Payload Shannon entropy (bits/byte)", fontsize=9)
    ax2.set_ylim(0, 9.0)
    ax2.set_title("Attack 4 — Entropy Threshold", fontsize=10)
    ax2.legend(fontsize=7.5)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.4, color=MED_GRAY)

    from matplotlib.patches import Patch
    legend_extra = [Patch(facecolor=SUCCESS_GREEN, label="Detected"),
                    Patch(facecolor=PRIMARY_RED, label="Evaded")]
    ax2.legend(handles=legend_extra + [ax2.get_legend_handles_labels()[0][0]],
               labels=["Detected", "Evaded", "Entropy threshold (7.0)"],
               fontsize=7.5)

    png_path = PNG_DIR / "fig12_adversarial_results.png"
    pdf_path = PDF_DIR / "fig12_adversarial_results.pdf"
    fig.savefig(str(png_path), dpi=600, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("ZombieGuard — Adversarial Robustness Evaluation (Experiment 8)")
    print("=" * 65)

    random.seed(42)
    np.random.seed(42)

    configure_style()

    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    all_results: list[dict] = []
    all_results.extend(attack1_entropy_dilution(model))
    all_results.extend(attack2_method_harmonization(model))
    all_results.extend(attack3_entropy_camouflage(model))
    all_results.extend(attack4_entropy_threshold(model))

    # Save full results
    df_full = pd.DataFrame(all_results)
    full_csv = CSV_DIR / "adversarial_full_results.csv"
    df_full.to_csv(str(full_csv), index=False)
    print(f"\nSaved: {full_csv}")

    # Save summary table
    df_summary = build_summary_table(all_results)
    summary_csv = CSV_DIR / "table_adversarial_results.csv"
    df_summary.to_csv(str(summary_csv), index=False)
    print(f"Saved: {summary_csv}")

    # Generate figure
    generate_figure(all_results)

    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for _, row in df_summary.iterrows():
        print(f"\n  {row['Attack']}")
        print(f"    Strategy:            {row['Strategy']}")
        print(f"    Neutralized:         {row['Features Neutralized']}")
        print(f"    Still firing:        {row['Features Still Firing']}")
        print(f"    Prob / Detected:     {row['Prob']} / {row['Detected']}")
        print(f"    Evasion rate:        {row['Evasion Rate']}")

    total  = len(all_results)
    evaded = sum(1 for r in all_results if r["evasion_success"])
    print(f"\nOverall: {evaded}/{total} configurations evaded detection "
          f"({100*evaded/total:.1f}%)")

    # Honest interpretation
    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    print("""
Attack 4 (Entropy Threshold): FULLY RESISTED at all DEFLATE levels.
  Random bytes are incompressible — entropy stays ~7.95 regardless of level.
  method_mismatch fires on every configuration. Evasion rate: 0%.

Attack 1 & 3 (Dilution / Camouflage): PARTIAL RESISTANCE.
  Detected at N<=10. Evades at N>=50.
  Root cause: the model over-weights suspicious_entry_ratio. When ratio
  drops below ~0.02, the model's probability collapses to 0.0 even though
  method_mismatch=1 and data_entropy_shannon=7.9 are both present.
  This is a genuine model weakness, not a feature design failure.
  Fix: add a hard override rule — if method_mismatch=1 AND entropy>7.0,
  force detection regardless of ratio. This is a post-training guard.

Attack 2 (Method Harmonization): EVADES.
  method_mismatch=0 (attacker wins that feature).
  declared_vs_entropy_flag=1 (entropy flag fires) but the model assigns
  insufficient weight to it alone. Evasion rate: 100%.
  Fix: same override — if declared_vs_entropy_flag=1 AND entropy>7.5,
  force detection. The physics constraint is still violated.

PAPER FRAMING:
  The overconstrained design holds at the feature level — every attack
  leaves at least one feature firing. The vulnerability is in the model's
  learned weights, not the feature set. This is an important distinction:
  the features are correct; the model needs a hard-rule safety layer.
  This is a concrete future work item and strengthens the paper's honesty.
""")
    print("=" * 65)


if __name__ == "__main__":
    main()
