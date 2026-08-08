# ZombieGuard model card

## Model summary

ZombieGuard is a ZIP structural-evasion detector. A bounded parser extracts byte-level
structural and entropy features, then a LightGBM binary classifier scores the archive.

| Property | Value |
|---|---|
| Task | ZIP metadata-evasion detection |
| Model family | LightGBM binary classifier |
| Output | Uncalibrated model score, verdict, and supporting reasons |
| Supported runtime | Python 3.11 and 3.12 |
| Supported format | ZIP only |
| Release model | `src/zombieguard/assets/zombieguard_lgbm.txt` |
| Metadata | `src/zombieguard/assets/zombieguard_lgbm.metadata.json` |
| Evaluation | `artifacts/metrics.json` |

## Intended use

The model is intended as a defensive triage signal for ZIP archives that may exploit
differences between archive parsers. Appropriate deployments include upload validation,
mail-gateway triage, sandbox pre-processing, and analyst tooling—always with an input-size
cap, external process controls, and additional content-scanning defenses.

It is not intended to:

- determine whether an archive or payload is malware;
- replace antivirus, sandboxing, content disarm, or manual review;
- scan RAR, 7z, APK, JAR, disk images, or arbitrary binary formats;
- make attribution, malware-family, prevalence, or temporal claims;
- justify automatically executing or extracting a file after a clean result.

## Inputs and behavior

The feature schema is versioned in the model metadata and enforced at load time. It covers
compression-method agreement, entropy, entry-level suspicious counts and ratios, CRC
consistency, and encryption state. The scanner parses archive structures directly; archive
contents are not executed.

Only successfully parsed archives are passed to the model. Invalid, incomplete,
unsupported, or input-size-limited archives produce an explicit unscannable result and
must not be represented as a zero-filled clean sample.

## Training and evaluation

The final release model is trained from the eligible synthetic-positive and benign-labeled
rows documented in the [data card](DATA_CARD.md). Its published benchmark uses eight outer
leave-one-attack-variant-out folds. Inside each outer training set, grouped inner
out-of-fold scores set a threshold against a fixed development 0.5% benign false-positive
budget. The untouched outer rows are used once for reporting and never participate in
their threshold. MalwareBazaar-derived and otherwise unverified positives are excluded
from both evaluation and release-model training.

The release model's threshold is calibrated from grouped out-of-fold scores across the
full development table using the same policy. LightGBM output is not probability
calibrated and must be described as a score. This operating-point policy is development
evidence, not an independent estimate of production false-positive rate.

Published values, including confusion counts, per-variant performance, Wilson 95%
confidence intervals, source-table hashes, leakage assertions, configuration, and model
checksum, live in [the metrics artifact](../artifacts/metrics.json). This card deliberately
does not restate rounded numbers that could drift from the signed-off artifact.

Reproduce or validate the release with:

```bash
python -m zombieguard.evaluate --check
python -m zombieguard.evaluate --train --check
```

`--check` verifies the artifact schema, data and model checksums, feature order, threshold
policy, leakage assertions, and a deterministic recomputation of the substantive metrics.
Training is seeded; native LightGBM serialization can still vary across library or platform
changes, so the committed model is checksum-verified rather than compared byte-for-byte
with a fresh retrain.

## Limitations and failure modes

- Evaluation uses synthetic positives and a mixed benign-labeled feature snapshot.
  Reported performance should not be generalized to live malware or unseen attack tooling.
- Synthetic-positive features were refreshed with the release parser. Benign rows are a
  legacy feature-only snapshot because their raw archives are unavailable for re-parsing.
- New structural evasions may not resemble any generated training variant.
- Encrypted entries, ZIP64, multi-disk archives, and unknown compression methods are
  reported as unsupported rather than scored clean. Streaming data descriptors are
  supported but remain an important parser-regression surface.
- Corrupt archives and ZIP bombs are adversarial inputs. The input-size cap reduces risk
  but cannot make in-process parsing completely safe; production deployments also need
  process-level memory and time limits.
- A valid, internally consistent archive can still contain malware and receive a clean
  structural result.
- Different downstream ZIP libraries can disagree on malformed input. Test against the
  parser used by the protected system when possible.

## Security and integrity

The package stores the LightGBM model as text rather than a Python pickle. The adjacent
metadata records its SHA-256 checksum and exact feature schema; loading must fail if either
does not match. This removes arbitrary-code execution through model deserialization but
does not make an untrusted package safe.

Run the scanner with least privilege and no network access where practical. Keep its
input-size cap enabled, and enforce memory and time limits around the process. See
[SECURITY.md](../SECURITY.md) for reporting and operational guidance.

## Versioning

A model change requires regenerated metadata and metrics. Changes to features, eligibility,
fold assignment, thresholding, or parser behavior are evaluation changes even when the
LightGBM parameters stay constant. Release notes should identify all such changes.
