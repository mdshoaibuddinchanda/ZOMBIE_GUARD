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

| Accuracy | Precision | Recall | F1     | ROC-AUC |
|:--------:|:---------:|:------:|:------:|:-------:|
|  0.9849  | 0.9962    | 0.9741 | 0.9850 | 0.9980  |

### Real-World Validation - 157 MalwareBazaar Samples

| Signal Type                                    | Count  | Share      |
|------------------------------------------------|-------:|-----------:|
| Gootloader-style EOCD chaining (EOCD > 1)      | 67     | 42.7%      |
| LFH/CDH method mismatch (CVE-2026-0866 class)  | 1      | 0.6%       |
| Undefined LFH method code                      | 1      | 0.6%       |
| **Total header evasion detected**              | **69** | **43.9%**  |
| Non-header-evasion ZIP malware (outside scope) | 88     | 56.1%      |

> ZombieGuard is a targeted header evasion detector, not a general
> malware scanner. The 88 clean files carry malware through other
> delivery mechanisms (macros, DLL injection) and are correctly
> outside ZombieGuard's detection scope.

## Project Structure

```text
# ZombieGuard

ZombieGuard is a machine learning system for detecting archive 
header evasion attacks in ZIP files, including CVE-2026-0866 
(Zombie ZIP) — where declared compression metadata conflicts 
with actual payload behavior to bypass antivirus scanning.

50 of 51 commercial antivirus engines miss this attack class.
ZombieGuard detects it using structural inconsistency analysis
and entropy-based payload inspection.

---

## Results

### Classifier Performance — XGBoost (Holdout Test Set)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|
| 0.9983 | 1.0000 | 0.9963 | 0.9981 | 0.9984 |

Confusion matrix: TN=329 FP=0 FN=1 TP=269

### Real-World Validation — MalwareBazaar

| Study | Samples | Detected | Rate | FP |
|---|---:|---:|---:|---:|
| Targeted (Gootloader family) | 165 | 77 | 46.7% | 0 |
| General (18 malware families) | 1366 | 93 | 6.8% | 0 |

**Key finding:** Archive header evasion is a specialist 
technique concentrated in specific families (~47% in 
Gootloader) rather than a universal approach across 
ZIP malware (~7% general prevalence). First published 
measurement of this attack class.

### Real-World Signal Breakdown (165 targeted samples)

| Signal Type | Count | Share |
|---|---:|---:|
| Gootloader-style EOCD chaining (EOCD > 1) | 66 | 85.7% |
| High entropy anomaly (new variant) | 7 | 9.1% |
| Undefined LFH method code (method=99) | 1 | 1.3% |
| True CVE-2026-0866 LFH/CDH mismatch | 1 | 1.3% |
| **Total detected** | **77** | **46.7%** |

> ZombieGuard is a targeted header evasion detector, not a 
> general malware scanner. Files carrying malware through 
> other mechanisms (macros, DLL injection) are correctly 
> outside its detection scope.

### SHAP Feature Importance

| Rank | Feature | SHAP Value |
|---:|---|---:|
| 1 | Renyi entropy of payload | 3.505 |
| 2 | Method mismatch (LFH vs CDH) | 3.162 |
| 3 | Shannon entropy of payload | 2.764 |
| 4 | LFH compression method | 0.964 |
| 5 | EOCD signature count | 0.500 |

Model learns the actual attack mechanism — not file size 
or format artifacts.

---

## What ZombieGuard Detects

Eight structural attack variants:

| Variant | Description |
|---|---|
| A — Classic Zombie ZIP | LFH declares STORE, CDH declares DEFLATE, payload compressed |
| B — Method-only mismatch | LFH and CDH method fields disagree, data stored |
| C — Gootloader concatenation | Multiple ZIPs chained, EOCD count > 1 |
| D — Multi-file decoy | Benign entries hide one malicious evasive entry |
| E — CRC32 mismatch | LFH CRC differs from CDH CRC |
| F — Extra field noise | Junk bytes in extra field shift parser offsets |
| G — High compression mismatch | Highly compressed payload with STORE declaration |
| H — Size field mismatch | LFH compressed size conflicts with CDH |

---

## How It Works

ZombieGuard extracts 12 structural and statistical features
from every archive file and scores it with XGBoost.

| Feature | Description |
|---|---|
| lf_compression_method | Method declared in Local File Header |
| cd_compression_method | Method declared in Central Directory |
| method_mismatch | LFH and CDH method fields disagree |
| data_entropy_shannon | Shannon entropy of payload bytes |
| data_entropy_renyi | Renyi entropy (alpha=2) of payload |
| declared_vs_entropy_flag | Claims STORE but payload looks compressed |
| eocd_count | Number of EOCD signatures (concatenation) |
| lf_unknown_method | Method code not in ZIP specification |
| suspicious_entry_count | Entries with LFH/CDH disagreement |
| suspicious_entry_ratio | Ratio of suspicious to total entries |
| any_crc_mismatch | CRC32 disagreement between LFH and CDH |
| is_encrypted | Encryption flag set in general purpose bits |

---

## Project Structure

```text
zombieguard/
├── src/
│   ├── extractor.py              # 12-feature ZIP binary parser
│   ├── classifier.py             # XGBoost training pipeline
│   ├── detector.py               # CLI detection tool
│   ├── entropy.py                # Shannon + Renyi entropy
│   ├── shap_analysis.py          # SHAP explainability figures
│   ├── transformer_model.py      # Byte-level Transformer (comparative)
│   ├── generalisation_study.py   # Cross-format study ZIP/APK/RAR/7z
│   ├── classifier_realworld.py   # Three-model comparison experiment
│   └── evaluate_hard_test.py     # Hard test set evaluation
├── data/
│   ├── generate_zombie_samples.py  # 8-variant synthetic generator
│   ├── collect_benign.py           # Benign sample collection
│   ├── download_realworld.py       # MalwareBazaar bulk downloader
│   ├── build_dataset.py            # Feature extraction pipeline
│   ├── split_realworld.py          # Train/val/test split
│   ├── build_hard_testset.py       # Hard evaluation set builder
│   ├── verify_realworld.py         # Real-world validation script
│   └── diagnose_features.py        # Feature distribution checker
├── tests/
│   ├── fixtures/                   # Test ZIP files
│   ├── test_extractor.py           # Extractor tests
│   └── test_classifier.py          # Pipeline tests
├── paper/figures/                  # SHAP plots + results tables
├── models/                         # Saved models (git-ignored)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/mdshoaibuddinchanda/zombieguard
cd ZOMBIE_GUARD
uv venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
uv pip install -r requirements.txt
```

---

## Usage

**Detect a single file:**
```bash
python src/detector.py suspicious.zip
```

**Verbose output showing all features:**
```bash
python src/detector.py suspicious.zip --verbose
```

**Batch scan a directory:**
```bash
python src/detector.py --batch /path/to/folder/
```

**Custom detection threshold:**
```bash
python src/detector.py suspicious.zip --threshold 0.3
```

---

## Reproduce the Dataset

```bash
# Generate 1348 synthetic malicious samples (8 variants)
python data/generate_zombie_samples.py

# Collect benign samples
python data/collect_benign.py

# Build feature dataset
python data/build_dataset.py

# Train the model
python src/classifier.py
```

**For real-world validation (requires MalwareBazaar API key):**
```bash
export MALWAREBAZAAR_API_KEY="your-key-here"
python data/download_realworld.py
python data/verify_realworld.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

Expected: 22 passed

---

## Generalisation Study

ZombieGuard was evaluated across four archive formats:

| Format | Recall | ROC-AUC | Notes |
|---|---:|---:|---|
| ZIP | 0.9963 | 0.9984 | Primary format |
| APK | 1.0000 | 1.0000 | ZIP-based, full transfer |
| RAR | 0.1400 | 0.9850 | AUC shows signal exists |
| 7z | 0.5950 | 1.0000 | Threshold calibration needed |

RAR and 7z AUC near 1.0 confirms entropy signals transfer
across formats. Low recall reflects threshold miscalibration
under distribution shift — not absence of signal.

---

## Research

Paper in preparation:
**"ZombieGuard: ML-Based Detection of Archive Header Evasion 
via Structural Inconsistency and Entropy Analysis"**

Target venue: Computers and Security (Elsevier)
Status: Writing

Figures and results tables: `paper/figures/`

---

## Security Note

- For authorized security research and defensive use only
- Real malware samples are excluded from this repository
- Never commit malware samples to version control
- `data/raw/`, `data/real_world_validation/` are git-ignored

---

## Citation

```bibtex
@software{zombieguard2026,
  author = {Md Shoaib Uddin Chanda},
  title  = {ZombieGuard: ML-Based Archive Header Evasion Detection},
  year   = {2026},
  url    = {https://github.com/mdshoaibuddinchanda/ZOMBIE_GUARD}
}
```
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
