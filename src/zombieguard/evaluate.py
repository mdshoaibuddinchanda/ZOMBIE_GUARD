"""Reproducible training, grouped evaluation, and artifact verification CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zombieguard.classifier import (
    ARTIFACT_FORMAT,
    EXTRACTOR_FEATURE_SCHEMA_VERSION,
    FEATURE_COLS,
    FEATURES_PATH,
    LABELS_PATH,
    MANIFEST_FILENAME,
    MODEL_PATH,
    SYNTHETIC_SCOPE,
    TARGET_ID,
    TrainingConfig,
    calibration_policy,
    load_model,
    metadata_path_for,
    save_model,
    sha256_file,
    train_and_evaluate,
)

METRICS_PATH = Path("artifacts/metrics.json")
MODEL_ARTIFACT_VERIFICATION_NOTE = (
    "The committed native model is schema-checked and SHA-256 verified. "
    "Dataset and evaluation results are deterministically recomputed; model text "
    "bytes are not compared with a retrain because LightGBM serialization may be "
    "platform-sensitive."
)


class ArtifactCheckError(ValueError):
    """Raised when committed evaluation artifacts are stale or inconsistent."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def train_artifacts(
    *,
    features_path: str | Path = FEATURES_PATH,
    labels_path: str | Path = LABELS_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and write a verified native model and metrics JSON."""

    result = train_and_evaluate(
        features_path,
        labels_path,
        TrainingConfig(random_state=random_state),
    )
    metrics = result["metrics"]
    release_threshold = result["release_threshold"]
    release_calibration = metrics["evaluation"]["release_calibration"]
    artifact = save_model(
        result["model"],
        model_path,
        threshold=release_threshold,
        metadata={
            "training_scope": SYNTHETIC_SCOPE,
            "training_dataset": metrics["dataset"],
            "evaluation_protocol": metrics["evaluation"]["protocol"],
            "calibration_policy": calibration_policy(),
            "release_calibration": release_calibration,
        },
    )
    metrics["model"] = {
        "artifact_format": ARTIFACT_FORMAT,
        "model_sha256": artifact["model_sha256"],
        "metadata_sha256": artifact["metadata_sha256"],
        "threshold": release_threshold,
        "calibration_policy": calibration_policy(),
        "training_rows": metrics["dataset"]["included_rows"],
        "verification_note": MODEL_ARTIFACT_VERIFICATION_NOTE,
    }
    _write_json(Path(metrics_path), metrics)
    return metrics


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ArtifactCheckError(f"required artifact is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCheckError(f"artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactCheckError(f"artifact root must be a JSON object: {path}")
    return payload


def _first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    """Return the first useful JSON-path difference for integrity errors."""

    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: key mismatch (missing={missing}, extra={extra})"
        for key in sorted(expected):
            difference = _first_difference(
                expected[key], actual[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: expected {len(expected)} items, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            difference = _first_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: expected {expected!r}, got {actual!r}"
    return None


def check_artifacts(
    *,
    features_path: str | Path = FEATURES_PATH,
    labels_path: str | Path = LABELS_PATH,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> dict[str, Any]:
    """Read-only verification of model, metadata, source hashes, and metrics."""

    features_path = Path(features_path)
    labels_path = Path(labels_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    # load_model verifies the native format, sidecar schema, feature order, and
    # model checksum before LightGBM parses the artifact.
    load_model(model_path)
    metadata = _read_json_object(metadata_path_for(model_path))
    metrics = _read_json_object(metrics_path)

    if metrics.get("scope") != SYNTHETIC_SCOPE:
        raise ArtifactCheckError(
            "metrics scope must be explicitly synthetic_positive_benchmark"
        )
    target = metrics.get("target")
    if not isinstance(target, dict) or target.get("id") != TARGET_ID:
        raise ArtifactCheckError("metrics target does not match classifier target")
    if metrics.get("feature_names") != FEATURE_COLS:
        raise ArtifactCheckError("metrics feature schema does not match code")
    if (
        metrics.get("extractor_feature_schema_version")
        != EXTRACTOR_FEATURE_SCHEMA_VERSION
    ):
        raise ArtifactCheckError(
            "metrics extractor/feature schema version does not match code"
        )

    dataset = metrics.get("dataset")
    if not isinstance(dataset, dict):
        raise ArtifactCheckError("metrics dataset section is missing")
    expected_source_hashes = {
        "features_csv_sha256": sha256_file(features_path),
        "labels_csv_sha256": sha256_file(labels_path),
    }
    manifest_path = features_path.parent / MANIFEST_FILENAME
    if manifest_path.exists():
        expected_source_hashes["dataset_manifest_sha256"] = sha256_file(
            manifest_path
        )
    for key, digest in expected_source_hashes.items():
        if dataset.get(key) != digest:
            raise ArtifactCheckError(
                f"metrics are stale: {key} does not match the current dataset"
            )

    evaluation = metrics.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ArtifactCheckError("metrics evaluation section is missing")
    if evaluation.get("protocol") != "nested_leave_one_generated_attack_variant_out":
        raise ArtifactCheckError("metrics use an unsupported evaluation protocol")
    if evaluation.get("leakage_check_passed") is not True:
        raise ArtifactCheckError("metrics do not attest a passed leakage check")
    if evaluation.get("out_of_fold_rows") != dataset.get("included_rows"):
        raise ArtifactCheckError("not every eligible row has an out-of-fold prediction")

    random_state = evaluation.get("random_state")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ArtifactCheckError("evaluation random_state must be an integer")
    if evaluation.get("decision_policy") != calibration_policy():
        raise ArtifactCheckError("metrics threshold-calibration policy does not match code")
    release_threshold = evaluation.get("release_threshold")
    if isinstance(release_threshold, bool) or not isinstance(
        release_threshold, (int, float)
    ):
        raise ArtifactCheckError("evaluation release threshold must be numeric")

    # Recompute every substantive claim.  The model artifact section is handled
    # separately below because native LightGBM text serialization can vary by
    # platform even when predictions and evaluation results are identical.
    recomputed = train_and_evaluate(
        features_path,
        labels_path,
        TrainingConfig(random_state=random_state),
    )["metrics"]
    reported_substantive = {
        key: value for key, value in metrics.items() if key != "model"
    }
    difference = _first_difference(recomputed, reported_substantive)
    if difference is not None:
        raise ArtifactCheckError(
            "reported dataset/evaluation metrics do not match deterministic "
            f"recomputation: {difference}"
        )
    folds = evaluation.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ArtifactCheckError("metrics do not contain grouped evaluation folds")
    held_out = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise ArtifactCheckError("invalid fold entry in metrics")
        if fold.get("attack_variant_overlap") != []:
            raise ArtifactCheckError("metrics contain attack-variant leakage")
        calibration = fold.get("calibration")
        if not isinstance(calibration, dict):
            raise ArtifactCheckError("metrics fold calibration is missing")
        if calibration.get("outer_test_rows_used") != 0:
            raise ArtifactCheckError("outer test rows were used for threshold calibration")
        if calibration.get("inner_coverage_complete") is not True:
            raise ArtifactCheckError("inner threshold calibration coverage is incomplete")
        if fold.get("threshold") != calibration.get("threshold"):
            raise ArtifactCheckError("fold threshold does not match its calibration")
        held_out.append(fold.get("held_out_attack_variant"))
    variant_counts = dataset.get("attack_variant_counts")
    if not isinstance(variant_counts, dict) or sorted(held_out) != sorted(variant_counts):
        raise ArtifactCheckError("folds do not cover every generated attack variant")

    model_section = metrics.get("model")
    if not isinstance(model_section, dict):
        raise ArtifactCheckError("metrics model section is missing")
    if model_section.get("artifact_format") != ARTIFACT_FORMAT:
        raise ArtifactCheckError("metrics describe an unexpected model format")
    if model_section.get("verification_note") != MODEL_ARTIFACT_VERIFICATION_NOTE:
        raise ArtifactCheckError("metrics omit the model verification limitation")
    if model_section.get("calibration_policy") != calibration_policy():
        raise ArtifactCheckError("metrics model calibration policy does not match code")
    model_digest = sha256_file(model_path)
    sidecar_digest = sha256_file(metadata_path_for(model_path))
    if model_section.get("model_sha256") != model_digest:
        raise ArtifactCheckError("metrics model checksum does not match the artifact")
    if model_section.get("metadata_sha256") != sidecar_digest:
        raise ArtifactCheckError("metrics metadata checksum does not match the sidecar")
    if metadata.get("model_sha256") != model_digest:
        raise ArtifactCheckError("sidecar model checksum does not match the artifact")
    if metadata.get("training_scope") != SYNTHETIC_SCOPE:
        raise ArtifactCheckError(
            "model sidecar does not declare synthetic-positive benchmark training"
        )
    if metadata.get("training_dataset") != dataset:
        raise ArtifactCheckError("model sidecar dataset lineage does not match metrics")
    if metadata.get("evaluation_protocol") != evaluation.get("protocol"):
        raise ArtifactCheckError("model sidecar evaluation protocol does not match metrics")
    if metadata.get("calibration_policy") != calibration_policy():
        raise ArtifactCheckError("model sidecar calibration policy does not match code")
    if metadata.get("release_calibration") != evaluation.get("release_calibration"):
        raise ArtifactCheckError("model sidecar release calibration does not match metrics")
    if metadata.get("threshold") != release_threshold:
        raise ArtifactCheckError("model sidecar threshold does not match metrics")
    if model_section.get("threshold") != release_threshold:
        raise ArtifactCheckError("metrics model threshold does not match evaluation")
    if model_section.get("training_rows") != dataset.get("included_rows"):
        raise ArtifactCheckError("metrics model training count does not match dataset")

    return {
        "model_sha256": model_digest,
        "included_rows": dataset["included_rows"],
        "attack_variants": len(variant_counts),
        "scope": metrics["scope"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and verify ZombieGuard's synthetic structural-evasion model"
        )
    )
    parser.add_argument("--train", action="store_true", help="write model and metrics")
    parser.add_argument(
        "--check", action="store_true", help="verify artifacts without modifying them"
    )
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=METRICS_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.train and not args.check:
        parser.error("select at least one of --train or --check")

    if args.train:
        metrics = train_artifacts(
            features_path=args.features,
            labels_path=args.labels,
            model_path=args.model_out,
            metrics_path=args.metrics_out,
            random_state=args.random_state,
        )
        aggregate = metrics["evaluation"]["aggregate_synthetic_metrics"]
        print(
            "Synthetic grouped evaluation: "
            f"accuracy={aggregate['accuracy']:.4f}, "
            f"precision={aggregate['precision']:.4f}, "
            f"recall={aggregate['recall']:.4f}, "
            f"f1={aggregate['f1']:.4f}, "
            f"roc_auc={aggregate['roc_auc']:.4f}"
        )
        print(f"Wrote model: {args.model_out}")
        print(f"Wrote metrics: {args.metrics_out}")
        print(
            "Release threshold: "
            f"{metrics['evaluation']['release_threshold']:.12g} "
            f"({metrics['evaluation']['decision_policy']['id']})"
        )

    if args.check:
        summary = check_artifacts(
            features_path=args.features,
            labels_path=args.labels,
            model_path=args.model_out,
            metrics_path=args.metrics_out,
        )
        print(
            "Artifact check passed: "
            f"{summary['included_rows']} rows, "
            f"{summary['attack_variants']} variants, "
            f"scope={summary['scope']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
