# ZombieGuard

ZombieGuard is an ML detector for archive header evasion, including
parser-confusion attacks such as CVE-2026-0866 style metadata/payload mismatch.

## Setup (UV)

```bash
git clone https://github.com/mdshoaibuddinchanda/zombieguard
cd zombieguard
uv venv .venv
# PowerShell
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## Reproduce Main Results

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

## Key Repro Outputs

- Table 1 source: `paper/figures/table1_baseline_comparison.csv`
- Table 4 source: `paper/figures/generalisation_results.csv`
- Table 5 source: `paper/figures/table5_feature_ablation.csv`
- Explainability: `paper/figures/shap_summary.png`,
  `paper/figures/shap_beeswarm.png`, `paper/figures/shap_waterfall.png`

## Artifact Policy

- Source-of-truth CSV files in `paper/figures/` are tracked.
- Generated PNG/PDF visual artifacts in `paper/figures/` are ignored to avoid
  output churn in commits.

## Security

- For authorized defensive research only.
- Malware binaries and large corpora are excluded from version control.

## Citation

```bibtex
@software{zombieguard2026,
  author = {Mohammed Shoaib Uddin Chanda},
  title  = {ZombieGuard: ML-Based Archive Header Evasion Detection},
  year   = {2026},
  url    = {https://github.com/mdshoaibuddinchanda/zombieguard}
}
```
