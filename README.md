# ZombieGuard

ZombieGuard is a machine-learning system for detecting archive header evasion patterns in ZIP containers, including the CVE-2026-0866 attack class where declared header metadata conflicts with actual payload behavior. Instead of acting as a general malware scanner, ZombieGuard focuses on structural and entropy inconsistencies that indicate parser-confusion techniques used to bypass static inspection pipelines.

## What ZombieGuard Detects

ZombieGuard detects archive header evasion attacks where ZIP metadata is intentionally manipulated.

1. Variant A - Classic Zombie ZIP: LFH declares STORE while CDH declares DEFLATE with compressed payload bytes.
2. Variant B - Method-only mismatch: LFH and CDH compression methods disagree while payload is stored.
3. Variant C - Gootloader concatenation: multiple ZIP containers concatenated to create EOCD-chain ambiguity.
4. Variant D - Multi-file decoy container: benign-looking entries hide one structurally evasive payload entry.
5. Variant E - CRC mismatch: LFH CRC value differs from CDH CRC to confuse validator trust paths.
6. Variant F - Extra-field noise: junk bytes in extra fields shift parser offsets and boundary assumptions.
7. Variant G - High-compression mismatch: highly compressed payload paired with misleading STORE declaration.
8. Variant H - Size-field mismatch: LFH compressed-size metadata conflicts with CDH/payload reality.

## Results

### XGBoost Classifier - Holdout Test Set (530 samples)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|
| 0.9849 | 0.9962 | 0.9741 | 0.9850 | 0.9980 |

### Real-World Validation - 157 MalwareBazaar Samples

| Signal Type | Count | Share |
|---|---:|---:|
| Gootloader-style EOCD chaining (EOCD > 1) | 67 | 42.7% |
| LFH/CDH method mismatch (CVE-2026-0866 class) | 1 | 0.6% |
| Undefined LFH method code | 1 | 0.6% |
| **Total header evasion detected** | **69** | **43.9%** |
| Non-header-evasion ZIP malware (outside scope) | 88 | 56.1% |

> ZombieGuard is a targeted header evasion detector, not a general
> malware scanner. The 88 clean files carry malware through other
> delivery mechanisms (macros, DLL injection) and are correctly
> outside ZombieGuard's detection scope.

## Project Structure

```text
zombieguard/
├── src/                        # Core detection, training, explainability, and study modules
│   ├── extractor.py            # ZIP structural + entropy feature extraction
│   ├── classifier.py           # XGBoost training/inference pipeline
│   ├── detector.py             # CLI detector (single-file and batch)
│   ├── entropy.py              # Shannon and Renyi entropy utilities
│   ├── shap_analysis.py        # SHAP analysis and figure generation
│   ├── transformer_model.py    # Byte-level Transformer baseline
│   └── generalisation_study.py # Cross-format generalization study
├── data/
│   ├── raw/                    # Raw benign/malicious ZIP collections (git-ignored)
│   ├── processed/              # features.csv and labels.csv
│   ├── real_world_validation/  # MalwareBazaar validation samples (git-ignored)
│   ├── build_dataset.py        # Dataset construction script
│   ├── generate_zombie_samples.py
│   ├── collect_benign.py
│   ├── download_malicious.py
│   └── download_realworld.py
├── tests/
│   ├── fixtures/               # Local test ZIP fixtures
│   ├── test_extractor.py
│   └── test_classifier.py
├── models/                     # Saved model artifacts (git-ignored)
├── paper/
│   └── figures/                # SHAP and generalization figures/tables
├── notebooks/                  # Analysis notebooks (kept clean)
├── requirements.txt            # Project dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/mdshoaibuddinchanda/zombieguard
cd zombieguard
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

1. Single file detection

```bash
python src/detector.py tests/fixtures/zombie.zip
```

1. Verbose detection showing features

```bash
python src/detector.py tests/fixtures/zombie.zip --verbose
```

1. Batch directory scan

```bash
python src/detector.py --batch data/real_world_validation/
```

## Dataset

ZombieGuard currently uses:

- 2647 total samples
- 1348 malicious ZIP samples across 8 structural variants
- 1299 benign ZIP samples from PyPI wheels and clean generated archives
- 157 real-world malicious ZIP samples from MalwareBazaar for prevalence validation

Regenerate from scratch:

```bash
python data/generate_zombie_samples.py
python data/collect_benign.py
python data/download_malicious.py
python data/build_dataset.py
```

## How It Works

ZombieGuard scores each archive using nine structural and statistical features extracted from ZIP internals.

| Feature                   | Description                                                     |
|---------------------------|-----------------------------------------------------------------|
| lf_compression_method     | Compression method declared in Local File Header (LFH)          |
| cd_compression_method     | Compression method declared in Central Directory Header (CDH)   |
| method_mismatch           | Whether LFH and CDH method fields disagree                      |
| data_entropy_shannon      | Shannon entropy of actual payload bytes                         |
| data_entropy_renyi        | Renyi entropy of payload bytes ($\alpha=2$)                     |
| declared_vs_entropy_flag  | LFH claims STORE but payload entropy looks compressed           |
| eocd_count                | Number of EOCD signatures (concatenation indicator)             |
| file_size_bytes           | File size metadata feature used in analysis exports             |
| lf_unknown_method         | LFH method code not in known ZIP method set                     |

Model interpretation artifacts are available in paper/figures/, including SHAP importance charts and format generalization figures.

## Research

A full paper is in preparation targeting IEEE Access and Computers & Security. Current publication assets, including SHAP plots and generalization charts, are in paper/figures/.

## Important Security Note

- This project is for authorized security research and defensive validation only.
- Real malware samples under data/real_world_validation/ are excluded from version control via .gitignore.
- Never commit malware samples or live payloads to any public repository.

## Citation

```bibtex
@software{zombieguard2026,
  author = {Md Shoaib Uddin Chanda},
  title  = {ZombieGuard: ML-Based Archive Header Evasion Detection},
  year   = {2026},
  url    = {https://github.com/mdshoaibuddinchanda/zombieguard}
}
```
