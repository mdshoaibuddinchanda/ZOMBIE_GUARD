# ZombieGuard

**A defensive scanner for structural metadata evasion in ZIP archives.**

[![CI](https://github.com/mdshoaibuddinchanda/zombieguard/actions/workflows/ci.yml/badge.svg)](https://github.com/mdshoaibuddinchanda/zombieguard/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Scope: ZIP](https://img.shields.io/badge/scope-ZIP%20archives-4c1.svg)](#scope-and-limitations)

ZombieGuard detects contradictions between ZIP local file headers, central directory
records, and payload characteristics. These contradictions can be used to make archive
contents appear empty or harmless to one parser while another parser still reaches the
payload.

It combines a bounded ZIP parser with a small LightGBM model. It does not extract or
execute archive contents.

> [!IMPORTANT]
> ZombieGuard detects archive-evasion signals, not malware. A clean result is not proof
> that the files inside an archive are safe.

## 60-second quickstart

ZombieGuard supports Python 3.11 and 3.12.

```bash
git clone https://github.com/mdshoaibuddinchanda/zombieguard.git
cd zombieguard
python -m pip install .
zombieguard scan suspicious.zip --include-features
```

The release model is installed with the package. Use `zombieguard scan --help` for the
input-size limit, JSON/SARIF output, and directory-scanning options.

For development, install the repository in editable mode:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## What it checks

ZombieGuard treats each ZIP as a consistency problem rather than looking for malware
signatures:

- local-file-header and central-directory agreement;
- compression method, size, offset, and CRC consistency;
- declared compression versus payload entropy;
- malformed, truncated, encrypted, and unsupported structures;
- suspicious behavior across all entries rather than only the first entry.

The scanner reports a verdict with supporting reason codes. Parsing failures are reported
as indeterminate or suspicious conditions; they are never silently converted into a clean
result.

## Architecture

```text
Untrusted ZIP bytes
       │
       ▼
Bounded structural parser ──► explicit parse/limit failures
       │
       ▼
Per-entry consistency features
       │
       ▼
Versioned LightGBM model
       │
       ▼
verdict + score + reasons
```

Structural validation errors produce an explicit unscannable result. Successfully parsed
archives are scored by the model, and observed inconsistencies are returned as reason
codes. Model metadata records the feature schema, training configuration, source hashes,
and model checksum.

## Verified evaluation

The release evaluation is intentionally narrow and reproducible. It does **not** claim
real-world malware detection or cross-format generalization.

| Property | Release protocol |
|---|---|
| Scope | Synthetic-positive ZIP structural-evasion feature benchmark |
| Eligible corpus | 1,150 generated positives and 1,568 deduplicated benign-labeled feature rows |
| Positive holdout | Leave one complete attack variant out per fold (8 variants) |
| Benign holdout | Deterministic SHA-256 partition; each sample is evaluated once |
| Decision threshold | Nested grouped calibration on outer-training rows only; 0.5% benign-FPR budget |
| Reported predictions | Outer-fold predictions only; held-out rows never set their threshold |
| Excluded from claims | MalwareBazaar-derived and otherwise unverified positive labels |
| Confidence | Wilson 95% intervals alongside aggregate metrics |

The current artifact reports:

| Metric | Result | 95% Wilson interval |
|---|---:|---:|
| Accuracy | 99.89% (2,715 / 2,718) | 99.68–99.96% |
| Precision | 99.74% | 99.24–99.91% |
| Recall | 100% (1,150 / 1,150) | 99.67–100% |
| F1 | 99.87% | — |
| ROC-AUC | 99.90% | — |
| False-positive rate | 0.191% (3 / 1,568) | 0.065–0.561% |

The confusion matrix is TN=1,565, FP=3, FN=0, TP=1,150. The authoritative results are
[artifacts/metrics.json](artifacts/metrics.json); that file also contains per-variant fold
summaries, data-source hashes, leakage assertions, model checksum, and exact configuration.
If a metric is used in a resume, paper, or presentation, quote its scope as **synthetic
nested leave-one-variant-out feature evaluation**.

The negative rows comprise 686 PyPI-derived, 478 generated hard-negative, 400 generated
small-benign, and 4 Office-derived feature rows. Negative source families are distributed
across folds rather than held out as families. The intervals describe row-level uncertainty
inside this benchmark; they do not estimate deployment uncertainty or domain shift.
LightGBM output is reported as an uncalibrated score, not a probability. The threshold
policy is a development operating point evaluated with nested folds; fresh benign archives
parsed by the current release are still required before treating its measured
false-positive rate as production evidence.

Why use synthetic positives? Public, independently verified examples of this narrow evasion
technique are scarce. Generation makes each structural mutation explicit and repeatable.
Synthetic performance is evidence that the detector recognizes those mutations; it is not
a substitute for an independently labeled real-world benchmark.

See the [data card](docs/DATA_CARD.md) and [model card](docs/MODEL_CARD.md) for provenance,
label semantics, and failure modes.

## Reproduce the release

Install the core dependencies and verify the committed artifacts. The extra generation and
refresh steps reproduce all safe synthetic positives and confirm that their committed
features still match the current parser before retraining:

```bash
python -m pip install -e .
python -m zombieguard.evaluate --check
python data/scripts/generate_zombie_samples.py
python scripts/refresh_synthetic_features.py
python -m zombieguard.evaluate --train --check
```

Training reads the committed feature and label tables and writes:

- `src/zombieguard/assets/zombieguard_lgbm.txt` — portable LightGBM model;
- `src/zombieguard/assets/zombieguard_lgbm.metadata.json` — schema and integrity metadata;
- `artifacts/metrics.json` — machine-readable evaluation report.

Continuous integration runs linting, dependency and static security auditing,
byte-compilation, tests with coverage, package builds on Python 3.11 and 3.12,
generator/parser alignment, committed artifact verification, and an isolated
train-and-check smoke test.

## Repository layout

```text
src/zombieguard/        installable scanner package and release assets
tests/                  self-contained unit, adversarial, and CLI tests
data/processed/         committed feature table and labels
data/scripts/           safe synthetic-data builders
scripts/                dataset curation and audit helpers
artifacts/metrics.json  reproducible release metrics
docs/                   data and model cards
```

Raw malware, downloaded corpora, credentials, and locally trained scratch models are not
part of the repository.

## Scope and limitations

- ZIP is the only supported archive format. RAR, 7z, APK, and JAR performance is not
  claimed.
- The model is trained on generated evasion variants and benign-labeled feature rows.
  Novel parser differentials may not match the learned feature distribution.
- Encrypted content cannot be inspected. Directory recursion discovers ZIP files in nested
  folders; it does not scan archives embedded inside other archives.
- Results should feed a quarantine or review workflow, not replace an antivirus engine,
  sandbox, content scanner, or human analyst.
- Adversarial archives can consume memory or CPU. Keep the `--max-size-mb` input cap
  enabled and add process-level memory and time limits in production deployments.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the parser, dataset, or evaluation
protocol. Do not commit malware samples, API credentials, or benchmark claims without a
reproducible artifact.

For a vulnerability or parser-bypass report, follow [SECURITY.md](SECURITY.md). Please do
not attach live malware to a public issue.

## License

Licensed under the [Apache License 2.0](LICENSE).

---

[Data card](docs/DATA_CARD.md) · [Model card](docs/MODEL_CARD.md) ·
[Security policy](SECURITY.md) · [Contributing](CONTRIBUTING.md) ·
[CI](https://github.com/mdshoaibuddinchanda/zombieguard/actions/workflows/ci.yml)
