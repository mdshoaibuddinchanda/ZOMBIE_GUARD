"""Create the reproducible ZombieGuard training snapshot.

The positive class is deliberately limited to deterministic ``zombie_*``
fixtures.  MalwareBazaar rows are excluded because malware membership is not
the same label as ZIP structural evasion, and the historical real-world labels
were model-generated rather than independently reviewed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
LABELS_PATH = ROOT / "data" / "processed" / "labels.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attack_variant(filename: str) -> str:
    match = re.match(r"zombie_([A-Z])_", filename)
    return match.group(1) if match else "unknown"


def curate() -> dict[str, object]:
    features = pd.read_csv(FEATURES_PATH)
    labels = pd.read_csv(LABELS_PATH)

    if not features["filename"].is_unique or not labels["filename"].is_unique:
        raise ValueError("Input filenames must be unique before curation")

    merged = features.merge(labels, on="filename", how="inner", validate="one_to_one")
    if len(merged) != len(features) or len(merged) != len(labels):
        raise ValueError("Feature and label files contain different filename sets")

    keep = (merged["label"] == 0) | (
        (merged["label"] == 1) & merged["filename"].str.startswith("zombie_")
    )
    curated = merged.loc[keep].copy()

    feature_columns = [
        column for column in features.columns if column != "filename"
    ]
    curated = curated.sort_values("filename").drop_duplicates(
        subset=feature_columns + ["label"], keep="first"
    )

    positives = curated[curated["label"] == 1]
    negatives = curated[curated["label"] == 0]
    if positives.empty or negatives.empty:
        raise ValueError("Curated data must contain both classes")
    if not positives["filename"].str.startswith("zombie_").all():
        raise ValueError("Positive class contains a non-synthetic row")

    output_features = curated[["filename", *feature_columns]]
    output_labels = curated[["filename", "label"]]
    output_features.to_csv(FEATURES_PATH, index=False, lineterminator="\n")
    output_labels.to_csv(LABELS_PATH, index=False, lineterminator="\n")

    variants = Counter(_attack_variant(name) for name in positives["filename"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "target": "zip_structural_evasion",
        "curation": "synthetic-positive-and-benign-feature-snapshot-v1",
        "samples": int(len(curated)),
        "positive_synthetic": int(len(positives)),
        "negative_benign": int(len(negatives)),
        "attack_variants": dict(sorted(variants.items())),
        "feature_columns": feature_columns,
        "features_sha256": _sha256(FEATURES_PATH),
        "labels_sha256": _sha256(LABELS_PATH),
        "limitations": [
            "The positive class is synthetic.",
            "No independently labelled real-positive benchmark is included.",
            "Metrics from this snapshot must be described as synthetic evaluation.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = curate()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
