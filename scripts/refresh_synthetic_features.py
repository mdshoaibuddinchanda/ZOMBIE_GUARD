"""Refresh generated-positive features with the current bounded ZIP parser.

This maintenance command never discovers or labels arbitrary samples.  The
positive filenames come exclusively from the existing curated labels CSV, all
must match ``zombie_*``, and the fixture directory must contain that exact set.
Benign rows are copied byte-for-byte at the CSV field level because their raw
archives are not part of the repository.

Safe default (read-only verification):

    python scripts/refresh_synthetic_features.py

Explicit refresh:

    python scripts/refresh_synthetic_features.py --write
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zombieguard.extractor import extract_features

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
LABELS_PATH = ROOT / "data" / "processed" / "labels.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.json"
SYNTHETIC_DIR = ROOT / "data" / "raw" / "malicious"
EXTRACTOR_PATH = ROOT / "src" / "zombieguard" / "extractor.py"

TARGET = "zip_structural_evasion"
EXTRACTOR_FEATURE_SCHEMA_VERSION = "2.0.0"
POSITIVE_NAME_RE = re.compile(r"^zombie_([A-Z])_.+_\d{4}\.zip$")


class RefreshError(ValueError):
    """Raised when a refresh would be unsafe, incomplete, or non-reproducible."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RefreshError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _unique_by_filename(
    rows: list[dict[str, str]], source: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = row.get("filename", "")
        if not filename:
            raise RefreshError(f"{source} contains an empty filename")
        if filename in indexed:
            raise RefreshError(f"{source} contains duplicate filename: {filename}")
        indexed[filename] = row
    return indexed


def _csv_bytes(fieldnames: list[str], rows: list[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _fixture_corpus_sha256(paths: list[Path]) -> str:
    """Hash ordered names and contents so the fixture corpus is auditable."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _feature_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        raise RefreshError("extractor returned an empty model feature")
    return str(value)


def build_refresh(
    *,
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
    manifest_path: Path = MANIFEST_PATH,
    synthetic_dir: Path = SYNTHETIC_DIR,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build prospective feature/manifest bytes without modifying the repository."""

    feature_fields, feature_rows = _read_csv_rows(features_path)
    label_fields, label_rows = _read_csv_rows(labels_path)
    if label_fields != ["filename", "label"]:
        raise RefreshError("labels CSV schema must be exactly: filename,label")
    if not feature_fields or feature_fields[0] != "filename":
        raise RefreshError("features CSV must begin with filename")
    feature_columns = feature_fields[1:]
    old_features = _unique_by_filename(feature_rows, "features CSV")
    labels = _unique_by_filename(label_rows, "labels CSV")
    if set(old_features) != set(labels):
        raise RefreshError("features and labels CSVs must have identical filename sets")

    invalid_labels = sorted(
        {
            row["label"]
            for row in label_rows
            if row.get("label") not in {"0", "1"}
        }
    )
    if invalid_labels:
        raise RefreshError(f"labels must contain only 0/1, found: {invalid_labels}")
    positive_names = {
        row["filename"] for row in label_rows if row["label"] == "1"
    }
    benign_names = {
        row["filename"] for row in label_rows if row["label"] == "0"
    }
    if not positive_names or not benign_names:
        raise RefreshError("curated labels must contain both classes")
    invalid_positive_names = sorted(
        name for name in positive_names if POSITIVE_NAME_RE.fullmatch(name) is None
    )
    if invalid_positive_names:
        raise RefreshError(
            "positive labels may only reference deterministic zombie_* fixtures: "
            + ", ".join(invalid_positive_names[:5])
        )

    fixture_paths = sorted(synthetic_dir.glob("*.zip"), key=lambda path: path.name)
    fixture_names = {path.name for path in fixture_paths}
    missing_fixtures = sorted(positive_names - fixture_names)
    unexpected_fixtures = sorted(fixture_names - positive_names)
    if missing_fixtures or unexpected_fixtures:
        details = []
        if missing_fixtures:
            details.append(f"missing={missing_fixtures[:5]}")
        if unexpected_fixtures:
            details.append(f"unexpected={unexpected_fixtures[:5]}")
        raise RefreshError(
            "synthetic fixture directory must exactly match positive labels ("
            + "; ".join(details)
            + ")"
        )

    refreshed: dict[str, dict[str, str]] = {}
    parse_failures: list[str] = []
    for fixture_path in fixture_paths:
        extracted = extract_features(fixture_path)
        if extracted.get("parse_status") != "ok":
            parse_failures.append(
                f"{fixture_path.name}: {extracted.get('parse_status')} "
                f"({extracted.get('parse_error')})"
            )
            continue
        missing_columns = [
            column for column in feature_columns if column not in extracted
        ]
        if missing_columns:
            raise RefreshError(
                f"extractor omitted features for {fixture_path.name}: {missing_columns}"
            )
        refreshed[fixture_path.name] = {
            "filename": fixture_path.name,
            **{
                column: _feature_text(extracted[column])
                for column in feature_columns
            },
        }
    if parse_failures:
        raise RefreshError(
            f"{len(parse_failures)} synthetic positives failed current parsing; "
            + " | ".join(parse_failures[:5])
        )

    # Follow label order and copy benign field strings unchanged.
    output_rows = [
        refreshed[row["filename"]]
        if row["filename"] in positive_names
        else old_features[row["filename"]]
        for row in label_rows
    ]
    feature_content = _csv_bytes(feature_fields, output_rows)

    variants = Counter(
        POSITIVE_NAME_RE.fullmatch(name).group(1)  # type: ignore[union-attr]
        for name in positive_names
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "target": TARGET,
        "extractor_feature_schema_version": EXTRACTOR_FEATURE_SCHEMA_VERSION,
        "curation": "current-parser-synthetic-positive-and-legacy-benign-snapshot-v2",
        "samples": len(label_rows),
        "positive_synthetic": len(positive_names),
        "negative_benign": len(benign_names),
        "attack_variants": dict(sorted(variants.items())),
        "feature_columns": feature_columns,
        "features_sha256": _sha256_bytes(feature_content),
        "labels_sha256": _sha256_file(labels_path),
        "extractor_sha256": _sha256_file(EXTRACTOR_PATH),
        "synthetic_fixture_corpus_sha256": _fixture_corpus_sha256(fixture_paths),
        "provenance": {
            "synthetic_positive_rows": (
                "Freshly extracted from deterministic inert zombie_* fixtures "
                "using the current bounded parser."
            ),
            "benign_rows": (
                "Legacy curated feature-only rows; raw benign archives are not "
                "present, so they cannot be re-extracted with the current parser."
            ),
        },
        "limitations": [
            "The positive class is synthetic.",
            (
                "The benign class contains legacy feature-only rows because "
                "the raw benign archives are absent."
            ),
            "No independently labelled real-positive benchmark is included.",
            "Metrics from this snapshot must be described as synthetic evaluation.",
        ],
    }
    return feature_content, _json_bytes(manifest), manifest


def refresh(*, write: bool = False) -> dict[str, Any]:
    """Check or explicitly write the deterministic generated-positive refresh."""

    feature_content, manifest_content, manifest = build_refresh()
    features_current = FEATURES_PATH.read_bytes() == feature_content
    manifest_current = (
        MANIFEST_PATH.exists() and MANIFEST_PATH.read_bytes() == manifest_content
    )
    if write:
        _atomic_write(FEATURES_PATH, feature_content)
        _atomic_write(MANIFEST_PATH, manifest_content)
        features_current = True
        manifest_current = True
    return {
        "features_current": features_current,
        "manifest_current": manifest_current,
        "wrote_files": write,
        "samples": manifest["samples"],
        "positive_synthetic": manifest["positive_synthetic"],
        "negative_benign": manifest["negative_benign"],
        "synthetic_fixture_corpus_sha256": manifest[
            "synthetic_fixture_corpus_sha256"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely refresh only curated zombie_* positive feature rows"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="atomically update features.csv and dataset_manifest.json",
    )
    args = parser.parse_args(argv)
    summary = refresh(write=args.write)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.write and not (
        summary["features_current"] and summary["manifest_current"]
    ):
        print("Synthetic-positive features are stale; rerun with --write.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
