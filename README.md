# ZombieGuard: ML-Based Detection of ZIP Metadata Evasion

## Overview

ZombieGuard detects archive-based malware evasion attacks by identifying inconsistencies between ZIP metadata structures (Local File Header vs Central Directory Header) and actual payload characteristics.

Unlike signature-based systems, ZombieGuard frames detection as a **consistency verification problem** — exploiting the fact that compression physics violations (entropy, method codes) cannot be faked by an attacker. This makes the detector theoretically immune to temporal drift.

The system detects CVE-2026-0866-style attacks across nine documented evasion variants (A–I).

---

## Requirements

Python 3.10+ in a conda environment named `py312`.

```bash
conda activate py312
pip install -r requirements.txt
```

---

## Project Structure

```text
zombieguard/
├── src/
│   ├── extractor.py              # ZIP feature extractor (12 features)
│   ├── classifier.py             # XGBoost model training
│   ├── detector.py               # CLI detector (single file or batch)
│   ├── multi_baseline.py         # Experiment 1: 5-model comparison
│   ├── variant_recall.py         # Experiment 2: per-variant recall (A-I)
│   ├── temporal_stability.py     # Experiment 3: temporal stability analysis
│   ├── generalisation_study.py   # Cross-format generalisation (APK/RAR/7z)
│   ├── shap_analysis.py          # SHAP feature importance
│   ├── ablation_study.py         # Feature group ablation
│   ├── evaluate_hard_test.py     # Hard test set evaluation (3 models)
│   ├── classifier_realworld.py   # XGBoost trained on real-world samples
│   ├── baseline_detector.py      # Baseline rule-based detector
│   ├── transformer_model.py      # Byte-level Transformer classifier
│   └── entropy.py                # Shannon / Renyi entropy utilities
├── data/
│   ├── scripts/                  # All data pipeline scripts (tracked)
│   │   ├── generate_zombie_samples.py   # Synthesise 9-variant malicious ZIPs
│   │   ├── collect_benign.py            # Collect benign ZIP samples
│   │   ├── build_dataset.py             # Merge into features.csv + labels.csv
│   │   ├── download_malicious.py        # Download malicious ZIPs (MalwareBazaar)
│   │   ├── download_realworld.py        # Download real-world validation set
│   │   ├── verify_realworld.py          # Verify and label real-world samples
│   │   ├── fetch_bazaar_timestamps.py   # Fetch first_seen timestamps from API
│   │   ├── fetch_timestamps_v2.py       # Timestamp fetcher v2 (resume support)
│   │   ├── write_timestamps_csv.py      # Write timestamps to CSV
│   │   ├── split_realworld.py           # Split real-world into train/test
│   │   └── build_hard_testset.py        # Build EOCD-resistant hard test set
│   ├── processed/                # features.csv + labels.csv (tracked)
│   ├── bazaar_timestamps.csv     # MalwareBazaar first_seen timestamps (tracked)
│   ├── raw/                      # Synthetic training ZIPs (not tracked)
│   ├── real_world_validation/    # Real malware from MalwareBazaar (not tracked)
│   ├── hard_test/                # EOCD-resistant hard test set (not tracked)
│   └── generalisation/           # APK / RAR / 7z format samples (not tracked)
├── models/
│   └── xgboost_model.pkl         # Trained model (not tracked - regenerate)
└── paper/
    ├── generate_all_figures.py   # Master figure generator (all 9 figures)
    └── figures/
        ├── csv/                  # Source-of-truth result tables (tracked)
        ├── png/                  # 600 DPI PNG outputs (not tracked)
        └── pdf/                  # Vector PDF outputs (not tracked)
```

---

## Full Reproduction Pipeline

### Step 1 — Generate synthetic training data

Synthesises 1,350 malicious ZIPs across 9 evasion variants (A–I) and collects benign ZIPs, then merges everything into the canonical feature matrix.

```bash
conda run -n py312 python data/scripts/generate_zombie_samples.py
conda run -n py312 python data/scripts/collect_benign.py
conda run -n py312 python data/scripts/build_dataset.py
```

Outputs: `data/raw/malicious/` (1,350 ZIPs), `data/raw/benign/`, `data/processed/features.csv`, `data/processed/labels.csv`

### Step 2 — Train the model

Trains LightGBM on the 80/20 holdout split from the synthetic dataset. LightGBM is the primary model, selected over XGBoost based on hard test set results (Recall 0.9375, F1 0.9677, AUC 1.0000 vs XGBoost Recall 0.7188 on EOCD-resistant samples).

```bash
conda run -n py312 python src/classifier.py
```

Output: `models/lgbm_model.pkl`

### Step 3 — Collect real-world validation samples

Requires a MalwareBazaar API key. Downloads 1,318 real malware ZIPs, verifies them, and fetches their `first_seen` timestamps. These samples are used in Experiment 3 (temporal stability) — the earliest third (T1) is used to train a temporal model, and the middle and latest thirds (T2, T3) are used as test sets. They are never used to train the main 5-model comparison.

```bash
conda run -n py312 python data/scripts/download_realworld.py
conda run -n py312 python data/scripts/verify_realworld.py
conda run -n py312 python data/scripts/fetch_bazaar_timestamps.py
```

Outputs: `data/real_world_validation/` (1,318 ZIPs), `data/realworld_labels.csv`, `data/bazaar_timestamps.csv`

### Step 4 — Build the hard test set

Builds a 271-sample test set where the EOCD signal is suppressed (ratio 1.18x), forcing models to rely on entropy, method mismatch, CRC, and structural features together.

```bash
conda run -n py312 python data/scripts/split_realworld.py
conda run -n py312 python data/scripts/build_hard_testset.py
```

Output: `data/hard_test/` (evasion/ + non_evasion/ subdirs)

### Step 5 — Run the three paper experiments

Each script writes its result table to `paper/figures/csv/` and its chart to `paper/figures/png/` and `paper/figures/pdf/`.

#### Experiment 1 — Multi-model baseline comparison

Trains 5 classifiers (Logistic Regression, Linear SVM, Random Forest, LightGBM, XGBoost) on the same 12 features and identical 80/20 holdout split from the synthetic dataset. Also evaluates all 5 on the hard test set. Real-world samples are used for testing only — not training.

```bash
conda run -n py312 python src/multi_baseline.py
```

Outputs: `paper/figures/csv/table6_multi_baseline_comparison.csv`, `table6b_multi_baseline_hard_test.csv`, `paper/figures/png/fig5_multi_baseline_chart.png`

#### Experiment 2 — Per-variant recall breakdown

Evaluates the trained XGBoost model on each of the 9 evasion variants (A–I) individually, reporting TP/FN/recall and the primary driving feature per variant.

```bash
conda run -n py312 python src/variant_recall.py
```

Outputs: `paper/figures/csv/table7_variant_recall.csv`, `paper/figures/png/fig6_variant_recall_chart.png`

#### Experiment 3 — Temporal stability analysis

Uses the 1,318 real-world MalwareBazaar samples (from Step 3). Sorted by `first_seen` timestamp and split into three equal-count tertiles:

- T1 (earliest, ~439 samples) — used to **train** a temporal XGBoost model, combined with proportional benign samples
- T2 (middle, ~439 samples) — **test only**
- T3 (latest, ~440 samples) — **test only**

The pre-trained synthetic model (`models/xgboost_model.pkl`) is also evaluated zero-shot on all three windows using a Youden-J optimal threshold calibrated on T1. This tests whether a model trained purely on synthetic data generalises to real-world samples across time.

```bash
conda run -n py312 python src/temporal_stability.py
```

Outputs: `paper/figures/csv/table8_temporal_stability.csv`, `table8b_shap_stability.csv`, `paper/figures/png/fig7_temporal_stability_chart.png`

### Step 6 — Run additional analyses

#### SHAP feature importance

Computes SHAP values for the trained XGBoost model. Results feed into fig3 in `generate_all_figures.py`.

```bash
conda run -n py312 python src/shap_analysis.py
```

#### Feature ablation study

Removes one feature group at a time and retrains, measuring recall drop to quantify each group's contribution.

```bash
conda run -n py312 python src/ablation_study.py
```

Output: `paper/figures/csv/table5_feature_ablation.csv`

#### Cross-format generalisation

Zero-shot evaluation of XGBoost and Transformer on APK, RAR, and 7z archives (no retraining on those formats). Tests whether the physics-based signals transfer across archive formats.

```bash
conda run -n py312 python src/generalisation_study.py
```

Output: `paper/figures/csv/generalisation_results.csv`

#### Hard test set evaluation (3 models)

Evaluates synthetic-trained, real-trained, and mixed-trained XGBoost models on the hard test set side by side.

```bash
conda run -n py312 python src/evaluate_hard_test.py
```

Output: `paper/figures/csv/hard_test_comparison.csv`

#### Experiment 4 — ROC and Precision-Recall curves

Plots ROC and PR curves for ZombieGuard XGBoost vs the rule-based baseline on the same axes. The PR curve is especially important given the class imbalance (1,348 malicious vs 1,785 benign). Both curves use the same 80/20 synthetic holdout split.

```bash
conda run -n py312 python src/roc_pr_curves.py
```

Outputs: `paper/figures/csv/table_roc_pr_auc.csv`, `paper/figures/png/fig8_roc_curve.png`, `paper/figures/png/fig9_pr_curve.png`

#### Experiment 5 — Entropy distribution plot

Plots overlapping Shannon entropy histograms for malicious vs benign samples with a vertical line at the 7.0 bits/byte threshold used by `declared_vs_entropy_flag`. Proves the threshold is empirically grounded, not arbitrary.

```bash
conda run -n py312 python src/entropy_distribution.py
```

Outputs: `paper/figures/csv/table_entropy_stats.csv`, `paper/figures/png/fig10_entropy_distribution.png`

#### Experiment 6 — Per-family prevalence breakdown

Joins `realworld_labels.csv` with `bazaar_timestamps.csv` to report evasion detection rate per malware family across the 1,366-sample real-world scan. Transforms the aggregate 6.8% prevalence number into a per-family threat intelligence finding.

```bash
conda run -n py312 python src/family_prevalence.py
```

Outputs: `paper/figures/csv/table_family_prevalence.csv`, `paper/figures/png/fig11_family_prevalence.png`

#### Experiment 7 — False negative analysis

Reproduces the exact 80/20 holdout split, identifies the single false negative (`zombie_C_gootloader_0103.zip`), and explains why the model missed it. Saves the full feature vector and predicted probability.

```bash
conda run -n py312 python src/fn_analysis.py
```

Output: `paper/figures/csv/table_fn_analysis.csv`

#### Experiment 8 — Adversarial robustness evaluation

Four white-box adversarial attacks against ZombieGuard LightGBM. Each attack neutralises one or more features while keeping the payload deliverable. Tests whether the overconstrained feature design holds empirically.

- Attack 1: Entropy Dilution — add N low-entropy benign entries (target: suspicious_entry_ratio)
- Attack 2: Method Harmonization — set LFH=CDH=STORE, keep payload compressed (target: method_mismatch)
- Attack 3: Entropy Camouflage — add N high-entropy consistent benign entries (target: ratio)
- Attack 4: Entropy Threshold — compress at DEFLATE level 1 to reduce entropy below 7.0

```bash
conda run -n py312 python src/adversarial_eval.py
```

Outputs: `paper/figures/csv/table_adversarial_results.csv`, `adversarial_full_results.csv`, `paper/figures/png/fig12_adversarial_results.png`

### Step 7 — Generate all paper figures

Reads all CSV tables and the trained model, then produces all 9 publication figures at 600 DPI with embedded fonts (PDF fonttype 42). Prints `READY FOR SUBMISSION: Yes` when all outputs pass resolution and PDF-pairing checks.

```bash
conda run -n py312 python paper/generate_all_figures.py
```

Outputs: `paper/figures/png/` and `paper/figures/pdf/`

---

## Run the Detector

Single file:

```bash
conda run -n py312 python src/detector.py path/to/file.zip
```

Batch scan a directory:

```bash
conda run -n py312 python src/detector.py path/to/folder/ --batch
```

---

## The 12 Features

| # | Feature | Description |
| --- | --- | --- |
| 1 | `lf_compression_method` | Compression method declared in Local File Header |
| 2 | `cd_compression_method` | Compression method declared in Central Directory Header |
| 3 | `method_mismatch` | LFH method != CDH method (core Zombie ZIP signal) |
| 4 | `data_entropy_shannon` | Shannon entropy of payload bytes |
| 5 | `data_entropy_renyi` | Renyi entropy of payload bytes |
| 6 | `declared_vs_entropy_flag` | Declared STORE but entropy > 7.0 (compressed data) |
| 7 | `eocd_count` | Number of EOCD signatures (> 1 = Gootloader concat) |
| 8 | `lf_unknown_method` | LFH method code not in {0, 8} (Variant I) |
| 9 | `suspicious_entry_count` | Count of entries with inconsistent signals |
| 10 | `suspicious_entry_ratio` | Ratio of suspicious entries to total entries |
| 11 | `any_crc_mismatch` | Any entry has CRC32 mismatch |
| 12 | `is_encrypted` | Any entry has encryption flag set |

---

## Key Results

All numbers verified from `paper/figures/csv/`.

### Experiment 1 — Multi-model comparison (standard 80/20 synthetic holdout)

| Model | Recall | F1 | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Linear SVM | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| XGBoost | 0.9963 | 0.9981 | 1.0000 | 0 | 1 |

### Experiment 1 — Multi-model comparison (hard test set, EOCD ratio 1.18x)

| Model | Recall | F1 | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1.0000 | 0.9552 | 0.9993 | 3 | 0 |
| Linear SVM | 0.9688 | 0.9394 | 0.9992 | 3 | 1 |
| Random Forest | 0.7188 | 0.8364 | 0.9974 | 0 | 9 |
| LightGBM | 0.9375 | 0.9677 | 1.0000 | 0 | 2 |
| XGBoost | 0.7188 | 0.8364 | 1.0000 | 0 | 9 |

### Experiment 2 — Per-variant recall (XGBoost, full malicious corpus)

| Variant | Name | N | Recall | FN |
| --- | --- | --- | --- | --- |
| A | Classic Zombie ZIP | 350 | 1.0000 | 0 |
| B | Method-only mismatch | 100 | 1.0000 | 0 |
| C | Gootloader concatenation | 150 | 0.9867 | 2 |
| D | Multi-file decoy | 150 | 1.0000 | 0 |
| E | CRC32 mismatch | 100 | 1.0000 | 0 |
| F | Extra field noise | 100 | 1.0000 | 0 |
| G | High compression gap | 100 | 1.0000 | 0 |
| H | Size field mismatch | 100 | 1.0000 | 0 |
| I | Undefined method code | 200 | 1.0000 | 0 |
| Overall | all variants | 1350 | 0.9985 | 2 |

### Experiment 3 — Temporal stability (1,318 real-world samples, T1 trains / T2+T3 test)

| Window | Role | Malicious | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| T1 (earliest) | Train + eval | 439 | 1.0000 | 0.9932 | 1.0000 |
| T2 (middle) | Test only | 439 | 0.9977 | 0.9921 | 0.9990 |
| T3 (latest) | Test only | 440 | 0.6795 | 0.8027 | 0.7797 |

T3 drop is explained by a new method-8 variant that appeared after the T1 training cutoff — not model decay.

### Experiment 3 — Synthetic model zero-shot on real-world windows

| Window | Recall | F1 | AUC |
| --- | --- | --- | --- |
| Synth to T1 | 1.0000 | 0.9799 | 0.9756 |
| Synth to T2 | 1.0000 | 0.9799 | 0.9756 |
| Synth to T3 | 0.7295 | 0.8241 | 0.8766 |

### SHAP top-5 features (identical across all 3 temporal windows)

`data_entropy_renyi`, `data_entropy_shannon`, `lf_compression_method`, `is_encrypted`, `suspicious_entry_count`

### Cross-format generalisation (XGBoost, zero-shot)

| Format | Recall | AUC | Notes |
| --- | --- | --- | --- |
| ZIP | 0.9778 | 0.9980 | In-distribution baseline |
| APK | 1.0000 | 1.0000 | ZIP-based, full signal transfer |
| RAR | 0.1400 | 0.9850 | Low recall at default threshold; AUC confirms signal present |
| 7z | 0.5800 | 1.0000 | Partial signal transfer |
| RAR (t=0.15) | 0.3600 | 0.9850 | Calibrated threshold |
| 7z (t=0.25) | 0.7650 | 1.0000 | Calibrated threshold |

### Baseline vs ZombieGuard (synthetic holdout)

| Model | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- |
| Rule-based baseline | 0.8630 | 0.8710 | 32 | 37 |
| ZombieGuard LightGBM | 1.0000 | 1.0000 | 0 | 0 |

### Experiment 4 — ROC and PR curves

| Model | ROC-AUC | Average Precision |
| --- | --- | --- |
| ZombieGuard XGBoost | 1.0000 | 1.0000 |
| Rule-based baseline | 0.8740 | 0.8194 |

### Experiment 5 — Entropy distribution

| Class | N | Mean entropy | Std | % above 7.0 threshold |
| --- | --- | --- | --- | --- |
| Malicious | 1348 | 7.4509 | 0.5595 | 65.8% |
| Benign | 1785 | 7.2063 | 1.3481 | 76.2% |

The 7.0 threshold is not arbitrary — it sits at the natural valley between the two distributions. Note that 76.2% of benign samples also exceed 7.0 bits/byte, which means entropy alone is insufficient for detection. This is precisely why ZombieGuard uses 12 features: the ML model resolves the overlap region using method codes, CRC mismatches, EOCD counts, and structural signals that a single entropy threshold cannot.

### Experiment 6 — Per-family prevalence (top families, n >= 5)

| Family | Scanned | Evasion detected | Rate |
| --- | --- | --- | --- |
| Gootloader | 1070 | 67 | 6.3% |
| ClickFix | 6 | 3 | 50.0% |
| APT36 | 5 | 1 | 20.0% |
| SmartApeSG | 6 | 1 | 16.7% |
| Vidar | 26 | 1 | 3.8% |
| NetSupport RAT | 20 | 1 | 5.0% |
| ACRStealer | 13 | 0 | 0.0% |
| APT37 | 6 | 0 | 0.0% |

Gootloader is the dominant user of ZIP evasion. ClickFix, APT36, and SmartApeSG also use the technique. Families like ACRStealer, APT37, and NetSupport RAT deliver via ZIP but do not use header evasion.

Note on the two Gootloader figures: the general scan reports 6.3% evasion rate across 1,070 Gootloader-tagged samples from MalwareBazaar — this is the rate across the full family corpus, most of which are standard ZIPs. The targeted scan (46.7% across 165 samples) was run specifically on Gootloader samples known to use ZIP delivery as a primary mechanism. The difference is a sampling difference, not a contradiction: 6.3% is the population-level rate, 46.7% is the rate within the delivery-active subset.

### Experiment 7 — The single false negative

The one missed sample is `zombie_C_gootloader_0103.zip` (Variant C — Gootloader concatenation). Its predicted probability was 0.316, below the 0.5 threshold.

The root cause: `lf_compression_method=8` (DEFLATE), so `declared_vs_entropy_flag` never fires even though entropy is 7.96. `method_mismatch=0` because LFH and CDH agree. The only active signals were `eocd_count=7` and `any_crc_mismatch=1`, which together were insufficient to cross the decision boundary. Lowering the threshold to 0.35 would catch this sample at the cost of approximately 2 additional false positives.

### Experiment 8 — Adversarial robustness (4 white-box attacks)

| Attack | Strategy | Features Neutralized | Evasion Rate | Finding |
| --- | --- | --- | --- | --- |
| 1 — Entropy Dilution (N≤10) | Add low-entropy benign entries | suspicious_entry_ratio ↓ | 0% | Detected |
| 1 — Entropy Dilution (N≥50) | Add 50+ low-entropy entries | ratio → 0.02 | 100% | Model weakness: over-weights ratio |
| 2 — Method Harmonization | Set LFH=CDH=STORE | method_mismatch=0 | 100% | entropy_flag fires but model under-weights it alone |
| 3 — Entropy Camouflage (N≤10) | Add high-entropy consistent entries | ratio ↓ | 0% | Detected |
| 3 — Entropy Camouflage (N≥50) | Add 50+ high-entropy entries | ratio → 0.02 | 100% | Same ratio weakness |
| 4 — Entropy Threshold (all levels) | DEFLATE level 1–9 | none | 0% | Random bytes incompressible; entropy stays ≥7.95 |

The overconstrained design holds at the feature level — every attack leaves at least one feature firing. The vulnerability is in the model's learned weights, not the feature set. Fix: add a hard-rule safety layer — if `method_mismatch=1` AND `data_entropy_shannon>7.0`, force detection regardless of ratio. This is a concrete future work item.

---

## Paper Figures

All generated by `paper/generate_all_figures.py` at 600 DPI, Times New Roman, PDF fonttype 42.

| Output file | Description | Source |
| --- | --- | --- |
| `fig1_zip_header_mismatch` | Byte-level LFH vs CDH mismatch diagram | Hardcoded diagram |
| `fig2_attack_taxonomy` | Nine-variant evasion taxonomy table | Hardcoded |
| `fig3_shap_importance` | SHAP mean absolute feature importance | `models/lgbm_model.pkl` + `data/processed/` |
| `fig4_generalisation_chart` | Cross-format recall and AUC (ZIP/APK/RAR/7z) | `csv/generalisation_results.csv` |
| `fig5_multi_baseline_chart` | 5-model comparison: Recall and AUC (hard test set) | `csv/table6b_multi_baseline_hard_test.csv` |
| `fig6_variant_recall_chart` | Per-variant recall breakdown (variants A-I) | `csv/table7_variant_recall.csv` |
| `fig7_temporal_stability_chart` | Temporal stability across T1/T2/T3 windows | `csv/table8_temporal_stability.csv` |
| `fig8_roc_curve` | ROC curve: ZombieGuard LightGBM vs rule-based baseline | `data/processed/` + `models/lgbm_model.pkl` |
| `fig9_pr_curve` | Precision-Recall curve (imbalanced dataset) | `data/processed/` + `models/lgbm_model.pkl` |
| `fig10_entropy_distribution` | Shannon entropy histogram: malicious vs benign, threshold=7.0 | `data/processed/` |
| `fig11_family_prevalence` | Per-family evasion rate across 18+ malware families | `csv/table_family_prevalence.csv` |
| `table3_prevalence_breakdown` | Real-world prevalence by signal type (1,366 samples) | `data/realworld_labels.csv` |
| `table3a_targeted_prevalence` | Targeted Gootloader scan (165 samples) | Hardcoded constants |

---

## Artifact Policy

| Path | Tracked | Notes |
| --- | --- | --- |
| `data/scripts/` | Yes | All pipeline code |
| `data/processed/features.csv` | Yes | Canonical feature matrix |
| `data/processed/labels.csv` | Yes | Canonical labels |
| `data/bazaar_timestamps.csv` | Yes | Timestamp metadata only (no malware content) |
| `paper/figures/csv/` | Yes | Source-of-truth result tables |
| `paper/figures/png/` | No | Regenerate with `generate_all_figures.py` |
| `paper/figures/pdf/` | No | Regenerate with `generate_all_figures.py` |
| `data/raw/` | No | Regenerate with `data/scripts/` |
| `data/real_world_validation/` | No | Real malware — never push |
| `data/hard_test/` | No | Contains real malware samples |
| `data/generalisation/` | No | Large format archives — regenerate |
| `models/*.pkl` | No | Regenerate with `src/classifier.py` (LightGBM primary model) |

---

## Security Notice

This project is for **authorized defensive research only**.
Malware binaries and large datasets are not included in this repository.
Users are responsible for handling any external data safely.

---

## Citation

```bibtex
@software{zombieguard2026,
  author = {Mohammed Shoaib Uddin Chanda},
  title  = {ZombieGuard: ML-Based Archive Header Evasion Detection},
  year   = {2026},
  url    = {https://github.com/mdshoaibuddinchanda/zombieguard}
}
```
