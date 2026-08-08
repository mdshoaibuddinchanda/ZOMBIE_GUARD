# ZombieGuard data card

## Summary

The release corpus represents one narrow task: distinguishing generated ZIP
metadata-evasion variants from benign-labeled ZIP feature rows. It is not a malware corpus
and cannot support claims about general malware detection.

The repository distributes derived feature rows and labels, not the underlying downloaded
archives. This reduces the risk of accidentally redistributing malware and keeps the
benchmark reviewable.

## Release files

| File | Purpose |
|---|---|
| `data/processed/features.csv` | Parsed structural and entropy features keyed by filename |
| `data/processed/labels.csv` | Binary task labels keyed by filename |
| `data/processed/dataset_manifest.json` | Counts, schema version, provenance, and source hashes |
| `data/scripts/generate_zombie_samples.py` | Deterministic synthetic-positive generator |
| `scripts/curate_dataset.py` | Eligibility, provenance, and leakage audit |
| `scripts/refresh_synthetic_features.py` | Current-parser positive-feature verification/refresh |
| `artifacts/metrics.json` | Counts, source hashes, fold summaries, and evaluation results |

The evaluator records SHA-256 hashes of the two input tables. Results from edited tables
must not be presented as release results until the artifact is regenerated.

## Label definition

- `1` — the archive is known to contain a generated or independently verified structural
  metadata-evasion condition relevant to the detector.
- `0` — the archive is intended as a benign, internally consistent comparison sample.

The label is about archive structure. It is not a claim that the content is malicious or
safe. A malware-family name, download source, or previous model prediction is not
sufficient evidence for label `1`.

## Release benchmark population

The curated release snapshot contains 2,718 eligible rows:

| Class | Count | Role |
|---|---:|---|
| Generated structural-evasion positives | 1,150 | Positive, grouped by 8 attack variants |
| Benign-labeled ZIP feature rows | 1,568 | Negative, deterministically assigned across folds |

During repository migration, curation removed 197 `bazaar_*` positive rows, one stray
`test_*` positive row, and feature-identical duplicates before writing the current
snapshot. The current metrics artifact therefore reports zero excluded rows: those records
are no longer present in its input tables, not newly accepted into the benchmark. The
loader still rejects names marked as real, Bazaar-derived, pseudo-labeled, or stray test
data if they are reintroduced. These controls prevent unverified labels and repeated
feature vectors from inflating reported performance.

The exact eligible/excluded counts and rejection reasons in `artifacts/metrics.json` take
precedence over this narrative if the dataset version changes.

## Sources and generation

Synthetic positives are created by controlled edits to ZIP structures. The eight release
variants exercise different header-method, size, CRC, extra-field, and multi-entry
inconsistencies. Generation is valuable because the intended mutation is known exactly and
can be repeated without distributing a live payload.

All 1,150 positive rows were regenerated through the current release parser. The manifest
binds the extractor source, feature-schema version, feature table, labels, and generated
fixture corpus with SHA-256 digests.

The negative rows comprise 686 PyPI-derived, 478 generated hard-negative, 400 generated
small-benign, and 4 Office-derived feature rows. The committed table does not preserve
enough raw provenance to make demographic, software-ecosystem, or long-term
representativeness claims. Treat it as a convenience benchmark, not a census of benign ZIP
traffic.

## Split and leakage controls

Evaluation uses leave-one-attack-variant-out folds:

1. A complete positive generator variant is withheld from training.
2. Benign rows are assigned deterministically from SHA-256 identifiers so every eligible
   negative receives exactly one out-of-fold prediction.
3. Every reported aggregate prediction is produced by a model that did not train on that
   row.
4. Each outer fold's decision threshold is calibrated only from grouped inner out-of-fold
   scores on that outer training set; the outer test rows used to report results are never
   used to choose their threshold.
5. The evaluator checks fold coverage, overlap, threshold provenance, feature schema,
   label validity, duplicate
   identifiers, and source hashes before writing metrics.

This design measures transfer to an unseen *generated mutation family*. It does not prove
transfer to a real attacker, a different ZIP parser, or future evasion techniques.
Benign source/generator families are distributed across folds rather than held out as
families, so the negative-side result does not establish transfer to a new benign source.

## Known limitations and bias

- Generated positives may contain artifacts that make them easier to distinguish than
  real evasions.
- The benign sample may underrepresent very large, encrypted, streaming, nested, signed,
  self-extracting, or uncommon-compression archives.
- Filename-based provenance is weaker than a full immutable sample manifest. Future data
  releases should add content hashes, source licenses, collection dates, generator
  versions, and independently reviewed structural labels per archive.
- Raw source archives are absent, so third parties can reproduce the model from features
  but cannot independently re-parse every original benign input from this repository
  alone.
- Synthetic-only confidence intervals quantify sampling uncertainty within this benchmark;
  they do not quantify real-world domain shift.

## Appropriate use

Use the corpus for regression tests, feature-ablation work, and scoped comparisons of ZIP
structural-evasion detectors. Do not use it to claim antivirus efficacy, malware-family
coverage, prevalence, temporal stability, or RAR/7z/APK/JAR performance.

## Maintenance

Any corpus update must preserve the previous artifact for comparison or clearly version
the new one. Changes require a leakage audit, regenerated hashes and metrics, and an update
to this card when counts, provenance, or intended use changes.
