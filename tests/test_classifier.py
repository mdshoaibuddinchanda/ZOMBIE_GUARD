"""Self-contained tests for the trustworthy model and evaluation layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

import zombieguard.classifier as classifier_module
from zombieguard.classifier import (
    CALIBRATION_POLICY_ID,
    EXTRACTOR_FEATURE_SCHEMA_VERSION,
    FEATURE_COLS,
    DatasetValidationError,
    ModelIntegrityError,
    TrainingConfig,
    iter_variant_holdout_splits,
    load_model,
    load_training_dataset,
    metadata_path_for,
    predict,
    save_model,
    train_and_evaluate,
)
from zombieguard.evaluate import ArtifactCheckError, check_artifacts, train_artifacts


class FixedScoreModel:
    """Small non-serialized test double accepted by the prediction adapter."""

    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.score, dtype=float)


def _feature_values(signal: int = 0, sample_number: int = 0) -> dict[str, float]:
    entropy = 7.7 if signal else 5.0 + (sample_number % 10) / 100.0
    return {
        "lf_compression_method": float(0 if signal else 8),
        "cd_compression_method": 8.0,
        "method_mismatch": float(signal),
        "data_entropy_shannon": entropy,
        "data_entropy_renyi": entropy - 0.05,
        "declared_vs_entropy_flag": float(signal),
        "eocd_count": 1.0,
        "lf_unknown_method": 0.0,
        "suspicious_entry_count": float(signal),
        "suspicious_entry_ratio": float(signal),
        "any_crc_mismatch": 0.0,
        "is_encrypted": 0.0,
    }


def _write_tiny_dataset(
    directory: Path,
    *,
    duplicate_label: bool = False,
    drop_feature: str | None = None,
) -> tuple[Path, Path]:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    variants = ["A_classic", "B_method_only", "C_crc_mismatch"]
    for variant_number, variant in enumerate(variants):
        for sample_number in range(4):
            filename = f"zombie_{variant}_{sample_number:04d}.zip"
            feature_rows.append(
                {
                    "filename": filename,
                    **_feature_values(1, variant_number * 4 + sample_number),
                }
            )
            label_rows.append({"filename": filename, "label": 1})

    for sample_number in range(18):
        filename = f"benign_fixture_{sample_number:04d}.zip"
        feature_rows.append(
            {"filename": filename, **_feature_values(0, sample_number)}
        )
        label_rows.append({"filename": filename, "label": 0})

    excluded = [
        ("bazaar_deadbeef.zip", 1),
        ("realworld_unreviewed.zip", 0),
        ("pseudo_labeled.zip", 0),
        ("test_one_sample.zip", 1),
    ]
    for sample_number, (filename, label) in enumerate(excluded, start=100):
        feature_rows.append(
            {"filename": filename, **_feature_values(label, sample_number)}
        )
        label_rows.append({"filename": filename, "label": label})

    if duplicate_label:
        label_rows.append(dict(label_rows[0]))
    feature_frame = pd.DataFrame(feature_rows)
    if drop_feature is not None:
        feature_frame = feature_frame.drop(columns=[drop_feature])
    features_path = directory / "features.csv"
    labels_path = directory / "labels.csv"
    feature_frame.to_csv(features_path, index=False)
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    return features_path, labels_path


def test_dataset_filter_has_narrow_target_and_excludes_untrusted_rows(
    tmp_path: Path,
) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path)
    prepared = load_training_dataset(features_path, labels_path)

    assert len(prepared.frame) == 30
    positives = prepared.frame.loc[prepared.frame["label"].eq(1), "filename"]
    assert positives.str.startswith("zombie_").all()
    assert set(prepared.frame["attack_variant"]) == {
        "benign",
        "A_classic",
        "B_method_only",
        "C_crc_mismatch",
    }
    assert not prepared.frame["filename"].str.match(
        r"^(?:bazaar|realworld|pseudo|test_one)"
    ).any()
    assert prepared.summary["synthetic_positive_rows"] == 12
    assert prepared.summary["benign_rows"] == 18
    assert prepared.summary["excluded"]["total_excluded_rows"] == 4


def test_dataset_rejects_duplicate_filenames(tmp_path: Path) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path, duplicate_label=True)
    with pytest.raises(DatasetValidationError, match="duplicate filenames"):
        load_training_dataset(features_path, labels_path)


def test_dataset_rejects_missing_feature_schema(tmp_path: Path) -> None:
    features_path, labels_path = _write_tiny_dataset(
        tmp_path, drop_feature="any_crc_mismatch"
    )
    with pytest.raises(DatasetValidationError, match="any_crc_mismatch"):
        load_training_dataset(features_path, labels_path)


def test_variant_splits_are_deterministic_complete_and_leak_free(
    tmp_path: Path,
) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path)
    prepared = load_training_dataset(features_path, labels_path)
    first = list(iter_variant_holdout_splits(prepared, random_state=17))
    second = list(iter_variant_holdout_splits(prepared, random_state=17))

    assert [split.held_out_variant for split in first] == [
        split.held_out_variant for split in second
    ]
    assert all(
        np.array_equal(left.test_indices, right.test_indices)
        for left, right in zip(first, second, strict=True)
    )
    all_test_indices = np.concatenate([split.test_indices for split in first])
    assert sorted(all_test_indices) == sorted(prepared.frame.index)
    for split in first:
        train_variants = set(
            prepared.frame.loc[
                split.train_indices, "attack_variant"
            ].loc[
                prepared.frame.loc[split.train_indices, "label"].eq(1)
            ]
        )
        test_positive_variants = set(
            prepared.frame.loc[
                split.test_indices, "attack_variant"
            ].loc[
                prepared.frame.loc[split.test_indices, "label"].eq(1)
            ]
        )
        assert split.held_out_variant not in train_variants
        assert test_positive_variants == {split.held_out_variant}


def test_grouped_evaluation_labels_metrics_as_synthetic(tmp_path: Path) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path)
    result = train_and_evaluate(
        features_path,
        labels_path,
        TrainingConfig(random_state=7),
    )
    metrics = result["metrics"]

    assert metrics["scope"] == "synthetic_positive_benchmark"
    assert (
        metrics["extractor_feature_schema_version"]
        == EXTRACTOR_FEATURE_SCHEMA_VERSION
    )
    assert metrics["target"]["id"] == "zip_structural_evasion"
    assert metrics["evaluation"]["leakage_check_passed"] is True
    assert metrics["evaluation"]["out_of_fold_rows"] == 30
    assert metrics["evaluation"]["decision_policy"]["id"] == CALIBRATION_POLICY_ID
    assert result["release_threshold"] == metrics["evaluation"]["release_threshold"]
    assert len(metrics["evaluation"]["folds"]) == 3
    assert all(
        fold["attack_variant_overlap"] == []
        for fold in metrics["evaluation"]["folds"]
    )
    assert all(
        fold["calibration"]["outer_test_rows_used"] == 0
        and fold["calibration"]["inner_coverage_complete"] is True
        and fold["calibration"]["inner_oof_rows"] == fold["train_rows"]
        for fold in metrics["evaluation"]["folds"]
    )
    assert "malware detection" in " ".join(metrics["limitations"])


def test_outer_calibration_never_receives_outer_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path)
    prepared = load_training_dataset(features_path, labels_path)
    outer_splits = list(iter_variant_holdout_splits(prepared, random_state=13))
    calibration_indices: list[set[int]] = []
    original = classifier_module._calibrate_grouped_oof_threshold

    def capture_training_rows(
        training_frame: pd.DataFrame, **kwargs: int
    ) -> tuple[float, dict[str, object]]:
        calibration_indices.append(set(training_frame.index))
        return original(training_frame, **kwargs)

    monkeypatch.setattr(
        classifier_module,
        "_calibrate_grouped_oof_threshold",
        capture_training_rows,
    )
    train_and_evaluate(
        features_path,
        labels_path,
        TrainingConfig(random_state=13),
    )

    assert len(calibration_indices) == len(outer_splits)
    for seen, split in zip(calibration_indices, outer_splits, strict=True):
        assert seen == set(split.train_indices)
        assert seen.isdisjoint(set(split.test_indices))


def _fit_tiny_native_model() -> LGBMClassifier:
    rows = []
    labels = []
    for sample_number in range(40):
        label = sample_number % 2
        rows.append(_feature_values(label, sample_number))
        labels.append(label)
    model = LGBMClassifier(
        n_estimators=8,
        min_child_samples=2,
        num_leaves=4,
        random_state=3,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(pd.DataFrame(rows, columns=FEATURE_COLS), labels)
    return model


def test_native_model_round_trip_has_verified_json_sidecar(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny_model.txt"
    artifact = save_model(
        _fit_tiny_native_model(),
        model_path,
        metadata={"training_scope": "unit_test"},
    )
    loaded = load_model(model_path)
    result = predict(loaded, _feature_values(1))

    assert result["label"] == 1
    assert result["target"] == "zip_structural_evasion"
    assert "HEADER_METHOD_MISMATCH" in result["reason_codes"]
    metadata = json.loads(metadata_path_for(model_path).read_text(encoding="utf-8"))
    assert metadata["artifact_format"] == "lightgbm_booster_text"
    assert metadata["feature_names"] == FEATURE_COLS
    assert (
        metadata["extractor_feature_schema_version"]
        == EXTRACTOR_FEATURE_SCHEMA_VERSION
    )
    assert metadata["model_sha256"] == artifact["model_sha256"]
    assert "created_utc" not in metadata


def test_load_rejects_tampered_native_model(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny_model.txt"
    save_model(_fit_tiny_native_model(), model_path)
    with model_path.open("a", encoding="utf-8") as handle:
        handle.write("\n# tampered\n")

    with pytest.raises(ModelIntegrityError, match="checksum"):
        load_model(model_path)


def test_load_rejects_pickle_without_deserializing(tmp_path: Path) -> None:
    model_path = tmp_path / "untrusted.pkl"
    model_path.write_bytes(b"not even a pickle")
    with pytest.raises(ModelIntegrityError, match="execute code"):
        load_model(model_path)


def test_predict_requires_complete_finite_schema() -> None:
    features = _feature_values(0)
    features.pop("eocd_count")
    with pytest.raises(ValueError, match="eocd_count"):
        predict(FixedScoreModel(0.1), features)

    features = _feature_values(0)
    features["data_entropy_shannon"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predict(FixedScoreModel(0.1), features)


def test_high_entropy_stored_data_has_no_unsafe_hard_override() -> None:
    features = _feature_values(0)
    features["lf_compression_method"] = 0.0
    features["cd_compression_method"] = 0.0
    features["data_entropy_shannon"] = 7.99
    features["data_entropy_renyi"] = 7.98
    result = predict(FixedScoreModel(0.1), features)

    assert result["label"] == 0
    assert result["score"] == 0.1


def test_generic_reason_covers_other_structural_inconsistency() -> None:
    features = _feature_values(0)
    features["suspicious_entry_count"] = 1.0
    features["suspicious_entry_ratio"] = 1.0

    result = predict(FixedScoreModel(0.1), features)

    assert result["reason_codes"] == ["STRUCTURAL_INCONSISTENCY"]


def test_artifact_check_recomputes_and_rejects_metric_or_policy_tampering(
    tmp_path: Path,
) -> None:
    features_path, labels_path = _write_tiny_dataset(tmp_path)
    model_path = tmp_path / "model.txt"
    metrics_path = tmp_path / "metrics.json"
    train_artifacts(
        features_path=features_path,
        labels_path=labels_path,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=11,
    )
    check_artifacts(
        features_path=features_path,
        labels_path=labels_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    original = json.loads(metrics_path.read_text(encoding="utf-8"))
    tampered_payloads = []

    inflated = json.loads(json.dumps(original))
    inflated["evaluation"]["aggregate_synthetic_metrics"]["accuracy"] = 0.123456
    tampered_payloads.append(inflated)

    changed_threshold = json.loads(json.dumps(original))
    changed_threshold["evaluation"]["release_threshold"] *= 2.0
    tampered_payloads.append(changed_threshold)

    changed_policy = json.loads(json.dumps(original))
    changed_policy["evaluation"]["decision_policy"]["id"] = "unverified_policy"
    tampered_payloads.append(changed_policy)

    for reported in tampered_payloads:
        metrics_path.write_text(
            json.dumps(reported, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ArtifactCheckError):
            check_artifacts(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                metrics_path=metrics_path,
            )


def test_committed_dataset_contains_only_generated_target_positives() -> None:
    prepared = load_training_dataset()
    positives = prepared.frame.loc[prepared.frame["label"].eq(1), "filename"]

    assert len(positives) == 1150
    assert positives.str.startswith("zombie_").all()
    assert len(prepared.summary["attack_variant_counts"]) == 8
    assert prepared.frame[FEATURE_COLS].notna().all().all()
