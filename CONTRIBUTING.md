# Contributing

Contributions are welcome, especially minimal parser regressions, independently labeled
evaluation data, and improvements to resource limits or diagnostics.

## Development setup

Use Python 3.11 or 3.12:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activate the environment using the command for your shell, then run the same checks as CI:

```bash
python -m ruff check src tests data/scripts scripts
python -m pip_audit --skip-editable
python -m bandit -r src/zombieguard -q
python -m compileall -q src tests data/scripts scripts
python -m pytest --cov=zombieguard.entropy --cov=zombieguard.extractor --cov=zombieguard.classifier --cov=zombieguard.detector
python -m build
python -m zombieguard.evaluate --check
```

## Change guidelines

Keep production changes focused on ZIP scanning. Experimental model families and other
archive formats should not enter the core dependency set or public claims without a
separate, reproducible benchmark.

Parser changes should include tests for:

- valid archives, including data descriptors and empty files;
- truncation, bad offsets, inconsistent sizes, and overlapping records;
- configured input-size limits and unusually large entry tables;
- byte sequences that resemble ZIP signatures inside payload data;
- explicit error behavior—an unreadable file must never become a clean feature vector.

## Data and benchmark policy

- Never commit live malware, downloaded corpora, passwords, or API keys.
- Safe ZIP fixtures must be generated in the test itself or by a reviewed deterministic
  fixture builder.
- `label=1` means a verified structural-evasion condition; it does not mean "malware."
- Record provenance and SHA-256 digests. Deduplicate before any split.
- Keep generator variants, content families, and derived copies in one split group.
- Exclude self-labeled data from evaluation. A model prediction is not ground truth.
- Report out-of-fold or untouched-holdout metrics with confusion counts and confidence
  intervals; do not select a threshold on the test set.
- Calibrate an operating threshold only inside the corresponding training split and record
  the policy, budget, coverage, and achieved training-side rate.

When changing data, features, split logic, or model parameters, regenerate and review both
`artifacts/metrics.json` and the model metadata. The evaluator must pass its integrity and
leakage checks.

## Pull-request checklist

- [ ] Tests cover the behavior and a realistic failure case.
- [ ] Lint, compilation, tests, build, and artifact checks pass locally.
- [ ] No archive corpus, credential, private path, or generated cache is staged.
- [ ] User-visible behavior and limitations are documented.
- [ ] New metrics state their dataset, split unit, sample counts, and confidence interval.
- [ ] The bounded parser and input-size limit remain fail closed.

Security vulnerabilities should follow [SECURITY.md](SECURITY.md), not the normal issue
workflow.
