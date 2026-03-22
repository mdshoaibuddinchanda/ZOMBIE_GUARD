# ZombieGuard: ML-Based Detection of ZIP Metadata Evasion

## Overview

ZombieGuard is a machine learning system designed to detect archive-based malware evasion attacks by identifying inconsistencies between ZIP metadata structures (e.g., Local File Header vs Central Directory Header) and actual payload characteristics.

Unlike traditional signature or rule-based systems, ZombieGuard models detection as a **consistency verification problem**, enabling robust identification of parser differential attacks and metadata/payload mismatches.

---

## Key Features

* Detection of ZIP metadata evasion (e.g., LFH/CDH inconsistencies)
* Machine learning–based classification using engineered structural features
* Robust performance against parser confusion attacks (e.g., CVE-2026-0866–style)
* Cross-format evaluation (ZIP → RAR / 7z generalization)
* Explainability via SHAP feature importance analysis
* Full reproducibility pipeline

---

## Requirements

* Python 3.10+
* UV package manager
* Windows or Linux (tested on Windows)

---

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/mdshoaibuddinchanda/zombieguard
cd zombieguard
uv venv .venv
```

Activate environment:

```bash
# PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

---

## Reproduce Main Results

Run the full pipeline:

```bash
uv run python data/generate_zombie_samples.py
uv run python data/collect_benign.py
uv run python data/build_dataset.py
uv run python src/classifier.py
uv run python src/baseline_detector.py
uv run python src/generalisation_study.py
uv run python src/shap_analysis.py
uv run python src/ablation_study.py
uv run python paper/generate_all_figures.py
```

---

## Expected Results

Running the pipeline should reproduce:

* ~99.6% recall on synthetic evasion samples
* 0 false positives on benign samples
* Cross-format generalization results (RAR, 7z)
* Feature importance insights via SHAP

---

## Key Reproducible Outputs

* **Table 1 (Baseline Comparison):**
  `paper/figures/table1_baseline_comparison.csv`

* **Cross-format Results (Table 4):**
  `paper/figures/generalisation_results.csv`

* **Feature Ablation (Table 5):**
  `paper/figures/table5_feature_ablation.csv`

* **Explainability Outputs:**

  * `paper/figures/shap_summary.png`
  * `paper/figures/shap_beeswarm.png`
  * `paper/figures/shap_waterfall.png`

---

## Project Structure

* `data/` — dataset generation and preprocessing
* `src/` — model training and evaluation
* `paper/` — scripts and outputs used to generate paper figures

---

## Artifact Policy

* Source-of-truth CSV files in `paper/figures/` are version-controlled
* Generated image artifacts (PNG/PDF) are excluded to avoid unnecessary repository churn

---

## Security Notice

This project is intended for **authorized defensive research only**.

* Malware binaries and large datasets are **not included**
* Users are responsible for handling any external data safely

---

## Citation

If you use this work, please cite:

```bibtex
@software{zombieguard2026,
  author = {Mohammed Shoaib Uddin Chanda},
  title  = {ZombieGuard: ML-Based Archive Header Evasion Detection},
  year   = {2026},
  url    = {https://github.com/mdshoaibuddinchanda/zombieguard},
  note   = {GitHub repository}
}
```
