"""Extract the reproducible structural-evasion feature snapshot.

Only generated files named ``zombie_*.zip`` may enter the positive class.
Real malware is intentionally not auto-labelled: malware and ZIP-header
evasion are different targets and require independent review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zombieguard.extractor import extract_features

ROOT = Path(__file__).resolve().parents[2]
POSITIVE_DIR = ROOT / "data" / "raw" / "malicious"
BENIGN_DIR = ROOT / "data" / "raw" / "benign"
OUTPUT_FEATURES = ROOT / "data" / "processed" / "features.csv"
OUTPUT_LABELS = ROOT / "data" / "processed" / "labels.csv"

EXPORT_FEATURES = [
    "lf_compression_method",
    "cd_compression_method",
    "method_mismatch",
    "data_entropy_shannon",
    "data_entropy_renyi",
    "declared_vs_entropy_flag",
    "eocd_count",
    "lf_unknown_method",
    "file_size_bytes",
    "entry_count",
    "suspicious_entry_count",
    "suspicious_entry_ratio",
    "entropy_variance",
    "lf_crc_valid",
    "any_crc_mismatch",
    "is_encrypted",
]


def process_directory(directory: Path, label: int) -> list[dict[str, object]]:
    paths = sorted(path for path in directory.rglob("*.zip") if path.is_file())
    if label == 1:
        unexpected = [path.name for path in paths if not path.name.startswith("zombie_")]
        if unexpected:
            raise ValueError(
                "Positive directory contains non-generated samples: "
                + ", ".join(unexpected[:5])
            )

    rows: list[dict[str, object]] = []
    for path in paths:
        features = extract_features(str(path))
        if features.get("parse_status") != "ok":
            raise ValueError(f"Could not parse {path}: {features.get('parse_error')}")
        row = {column: features[column] for column in EXPORT_FEATURES}
        row["filename"] = path.relative_to(directory).as_posix()
        row["label"] = label
        rows.append(row)
    return rows


def main() -> None:
    rows = [
        *process_directory(POSITIVE_DIR, label=1),
        *process_directory(BENIGN_DIR, label=0),
    ]
    if not rows:
        raise RuntimeError("No ZIP samples found. Generate or provide safe fixtures first.")

    frame = pd.DataFrame(rows).sort_values("filename")
    if not frame["filename"].is_unique:
        raise ValueError("Dataset contains duplicate filenames")
    if set(frame["label"].astype(int)) != {0, 1}:
        raise RuntimeError(
            "Refusing to overwrite the release dataset without both positive and benign rows"
        )

    OUTPUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feature_tmp = OUTPUT_FEATURES.with_name(f".{OUTPUT_FEATURES.name}.tmp")
    label_tmp = OUTPUT_LABELS.with_name(f".{OUTPUT_LABELS.name}.tmp")
    try:
        frame[["filename", *EXPORT_FEATURES]].to_csv(
            feature_tmp, index=False, lineterminator="\n"
        )
        frame[["filename", "label"]].to_csv(
            label_tmp, index=False, lineterminator="\n"
        )
        feature_tmp.replace(OUTPUT_FEATURES)
        label_tmp.replace(OUTPUT_LABELS)
    finally:
        feature_tmp.unlink(missing_ok=True)
        label_tmp.unlink(missing_ok=True)
    print(
        f"Wrote {len(frame)} rows "
        f"({int((frame['label'] == 1).sum())} positive, "
        f"{int((frame['label'] == 0).sum())} benign)"
    )


if __name__ == "__main__":
    main()
