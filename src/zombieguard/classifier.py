"""Trustworthy training and inference for ZombieGuard.

The classifier has one deliberately narrow target: structural ZIP metadata
evasion.  A positive prediction means that ZIP structures resemble generated
archives whose local/central-directory metadata was intentionally made
inconsistent or corrupt.  It does *not* mean that the payload is malware.

Only generated ``zombie_*`` positives and label-0 benign-labeled feature rows are eligible
for training.  MalwareBazaar and other pseudo/real-world labels are excluded
because they do not independently establish structural evasion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES_PATH = Path("data/processed/features.csv")
LABELS_PATH = Path("data/processed/labels.csv")
MANIFEST_FILENAME = "dataset_manifest.json"
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "zombieguard_lgbm.txt"
EXTRACTOR_SOURCE_PATH = Path(__file__).resolve().with_name("extractor.py")
METADATA_SUFFIX = ".metadata.json"

TARGET_ID = "zip_structural_evasion"
TARGET_DEFINITION = (
    "Detection of deliberately inconsistent or corrupted ZIP local-header, "
    "central-directory, or end-of-central-directory metadata intended to hide "
    "archive entries from archive-aware scanners. A positive label does not "
    "assert that the payload is malware."
)
SYNTHETIC_SCOPE = "synthetic_positive_benchmark"
ARTIFACT_FORMAT = "lightgbm_booster_text"
ARTIFACT_SCHEMA_VERSION = 1
EXTRACTOR_FEATURE_SCHEMA_VERSION = "2.0.0"
DEFAULT_THRESHOLD = 0.5
CALIBRATION_POLICY_ID = "nested_grouped_oof_benign_quantile_v1"
BENIGN_FPR_BUDGET = 0.005
CALIBRATION_QUANTILE = 1.0 - BENIGN_FPR_BUDGET

FEATURE_COLS = [
    "lf_compression_method",
    "cd_compression_method",
    "method_mismatch",
    "data_entropy_shannon",
    "data_entropy_renyi",
    "declared_vs_entropy_flag",
    "eocd_count",
    "lf_unknown_method",
    "suspicious_entry_count",
    "suspicious_entry_ratio",
    "any_crc_mismatch",
    "is_encrypted",
]

_ZOMBIE_NAME_RE = re.compile(
    r"^zombie_(?P<variant>.+)_(?P<sample_id>\d+)\.zip$", re.IGNORECASE
)
_UNTRUSTED_SOURCE_RE = re.compile(
    r"^(?:bazaar|real(?:world)?|pseudo(?:label(?:ed)?)?)[_-]", re.IGNORECASE
)
_STRAY_TEST_NAMES = {"test.zip", "test_one_sample.zip"}


class DatasetValidationError(ValueError):
    """Raised when training inputs cannot support a trustworthy evaluation."""


class ModelIntegrityError(ValueError):
    """Raised when a model artifact or its metadata fails verification."""


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic model and evaluation settings."""

    random_state: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer")


@dataclass
class PreparedDataset:
    """Validated rows plus auditable source and filtering metadata."""

    frame: pd.DataFrame
    summary: dict[str, Any]
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class VariantSplit:
    """One leave-one-generated-variant-out train/test partition."""

    held_out_variant: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_path_for(model_path: str | Path) -> Path:
    """Return the required sidecar path for a native model artifact."""

    path = Path(model_path)
    return path.with_suffix(METADATA_SUFFIX)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _validate_filename_column(frame: pd.DataFrame, source: str) -> None:
    if "filename" not in frame.columns:
        raise DatasetValidationError(f"{source} is missing required column: filename")
    if frame["filename"].isna().any():
        raise DatasetValidationError(f"{source} contains an empty filename")
    names = frame["filename"].astype(str)
    if names.str.strip().eq("").any():
        raise DatasetValidationError(f"{source} contains an empty filename")
    duplicates = sorted(names[names.duplicated(keep=False)].unique())
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise DatasetValidationError(
            f"{source} contains duplicate filenames ({preview}); filenames must be unique"
        )


def _coerce_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLS if column not in frame.columns]
    if missing:
        raise DatasetValidationError(
            "features CSV is missing required model columns: " + ", ".join(missing)
        )

    result = frame[FEATURE_COLS].copy()
    boolean_text = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    }
    for column in FEATURE_COLS:
        series = result[column]
        if series.dtype == object:
            normalized = series.map(
                lambda value: boolean_text.get(value.strip().lower(), value)
                if isinstance(value, str)
                else value
            )
            try:
                result[column] = pd.to_numeric(normalized, errors="raise")
            except (TypeError, ValueError) as exc:
                raise DatasetValidationError(
                    f"feature column {column!r} contains a non-numeric value"
                ) from exc
        else:
            result[column] = pd.to_numeric(series, errors="raise")

    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_columns = [
            column
            for column in FEATURE_COLS
            if not np.isfinite(result[column].to_numpy(dtype=float)).all()
        ]
        raise DatasetValidationError(
            "feature values must be finite; invalid columns: " + ", ".join(bad_columns)
        )
    return result.astype(float)


def _attack_variant(filename: str) -> str:
    match = _ZOMBIE_NAME_RE.fullmatch(filename)
    if match is None:
        raise DatasetValidationError(
            f"synthetic positive filename does not identify a variant: {filename}"
        )
    return match.group("variant")


def _validate_optional_manifest(
    manifest_path: Path,
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    prepared: pd.DataFrame,
    source_hashes: Mapping[str, str],
) -> str | None:
    """Validate a colocated curated-dataset manifest when one is present."""

    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"dataset manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError("dataset manifest root must be a JSON object")
    checks = {
        "schema_version": 1,
        "target": TARGET_ID,
        "extractor_feature_schema_version": EXTRACTOR_FEATURE_SCHEMA_VERSION,
        "features_sha256": source_hashes["features_csv_sha256"],
        "labels_sha256": source_hashes["labels_csv_sha256"],
        "extractor_sha256": sha256_file(EXTRACTOR_SOURCE_PATH),
        "samples": int(len(prepared)),
        "positive_synthetic": int(prepared["label"].eq(1).sum()),
        "negative_benign": int(prepared["label"].eq(0).sum()),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise DatasetValidationError(
                f"dataset manifest field {key!r} is stale or invalid"
            )
    expected_columns = [
        column for column in features.columns if column != "filename"
    ]
    if manifest.get("feature_columns") != expected_columns:
        raise DatasetValidationError(
            "dataset manifest feature_columns do not match the features CSV"
        )
    manifest_variants = manifest.get("attack_variants")
    if not isinstance(manifest_variants, dict):
        raise DatasetValidationError("dataset manifest attack_variants is missing")
    actual_variants = (
        prepared.loc[prepared["label"].eq(1), "attack_variant"]
        .astype(str)
        .str.split("_", n=1)
        .str[0]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    if manifest_variants != actual_variants:
        raise DatasetValidationError(
            "dataset manifest attack-variant counts do not match curated labels"
        )
    if len(features) != len(labels):
        raise DatasetValidationError(
            "curated manifest requires one feature row for every label row"
        )
    return sha256_file(manifest_path)


def load_training_dataset(
    features_path: str | Path = FEATURES_PATH,
    labels_path: str | Path = LABELS_PATH,
) -> PreparedDataset:
    """Load, validate, and narrowly filter the model-development dataset.

    Eligible positive rows must have label ``1`` and a filename beginning with
    ``zombie_``.  Eligible negatives have label ``0`` and must not identify a
    Bazaar, real-world, pseudo-labeled, or stray test source.  Filtering happens
    before the merge, and every eligible label must have exactly one feature row.
    """

    features_path = Path(features_path)
    labels_path = Path(labels_path)
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    _validate_filename_column(features, "features CSV")
    _validate_filename_column(labels, "labels CSV")
    if "label" not in labels.columns:
        raise DatasetValidationError("labels CSV is missing required column: label")

    try:
        numeric_labels = pd.to_numeric(labels["label"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError("labels must be integers 0 or 1") from exc
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise DatasetValidationError("labels must contain only integers 0 and 1")
    labels = labels.assign(label=numeric_labels.astype(int))

    names = labels["filename"].astype(str)
    untrusted_source = names.str.match(_UNTRUSTED_SOURCE_RE)
    stray_test = names.str.lower().isin(_STRAY_TEST_NAMES)
    generated_positive = labels["label"].eq(1) & names.str.startswith("zombie_")
    trusted_benign = (
        labels["label"].eq(0) & ~untrusted_source & ~stray_test
    )
    eligible_mask = generated_positive | trusted_benign
    eligible_labels = labels.loc[eligible_mask, ["filename", "label"]].copy()

    feature_names = set(features["filename"].astype(str))
    missing_features = sorted(
        set(eligible_labels["filename"].astype(str)).difference(feature_names)
    )
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise DatasetValidationError(
            f"{len(missing_features)} eligible labels have no feature row ({preview})"
        )

    merged = eligible_labels.merge(
        features,
        on="filename",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    coerced = _coerce_features(merged)
    merged = pd.concat(
        [merged.drop(columns=FEATURE_COLS), coerced],
        axis=1,
    )
    merged["attack_variant"] = "benign"
    positive_mask = merged["label"].eq(1)
    merged.loc[positive_mask, "attack_variant"] = merged.loc[
        positive_mask, "filename"
    ].map(_attack_variant)

    class_counts = merged["label"].value_counts().to_dict()
    if set(class_counts) != {0, 1}:
        raise DatasetValidationError("eligible dataset must contain both classes")
    variants = sorted(
        merged.loc[positive_mask, "attack_variant"].astype(str).unique()
    )
    if len(variants) < 3:
        raise DatasetValidationError(
            "at least three generated attack variants are required for nested grouped evaluation"
        )

    excluded_labels = labels.loc[~eligible_mask]
    excluded_counts = {
        "untrusted_source_rows": int(untrusted_source.sum()),
        "stray_test_rows": int((stray_test & ~untrusted_source).sum()),
        "other_non_target_positive_rows": int(
            (
                labels["label"].eq(1)
                & ~generated_positive
                & ~untrusted_source
                & ~stray_test
            ).sum()
        ),
        "total_excluded_rows": int(len(excluded_labels)),
    }
    variant_counts = (
        merged.loc[positive_mask, "attack_variant"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    benign_names = merged.loc[merged["label"].eq(0), "filename"].astype(str)
    benign_composition = {
        "hard_negative": int(benign_names.str.match(r"^hard_negatives?_").sum()),
        "office_sample": int(benign_names.str.match(r"^office_sample_").sum()),
        "pypi_archive": int(benign_names.str.match(r"^pypi_").sum()),
        "small_benign": int(benign_names.str.match(r"^small_benign_").sum()),
    }
    classified_benign = sum(benign_composition.values())
    if classified_benign != int(class_counts[0]):
        benign_composition["other"] = int(class_counts[0]) - classified_benign
    summary: dict[str, Any] = {
        "raw_feature_rows": int(len(features)),
        "raw_label_rows": int(len(labels)),
        "included_rows": int(len(merged)),
        "synthetic_positive_rows": int(class_counts[1]),
        "benign_rows": int(class_counts[0]),
        "benign_feature_row_composition": benign_composition,
        "excluded": excluded_counts,
        "attack_variant_counts": variant_counts,
    }
    source_hashes = {
        "features_csv_sha256": sha256_file(features_path),
        "labels_csv_sha256": sha256_file(labels_path),
    }
    manifest_digest = _validate_optional_manifest(
        features_path.parent / MANIFEST_FILENAME,
        features=features,
        labels=labels,
        prepared=merged,
        source_hashes=source_hashes,
    )
    if manifest_digest is not None:
        source_hashes["dataset_manifest_sha256"] = manifest_digest

    return PreparedDataset(
        frame=merged[["filename", "label", "attack_variant", *FEATURE_COLS]].copy(),
        summary=summary,
        source_sha256=source_hashes,
    )


def _stable_benign_assignments(
    filenames: Sequence[str], variants: Sequence[str], random_state: int
) -> dict[str, str]:
    """Assign each benign sample exactly once, evenly and order-independently."""

    ranked = sorted(
        filenames,
        key=lambda name: hashlib.sha256(
            f"{random_state}:{name}".encode()
        ).digest(),
    )
    return {
        filename: variants[position % len(variants)]
        for position, filename in enumerate(ranked)
    }


def iter_variant_holdout_splits(
    dataset: PreparedDataset | pd.DataFrame,
    random_state: int = 42,
) -> Iterator[VariantSplit]:
    """Yield deterministic leave-one-generated-attack-variant-out splits.

    Every positive variant is wholly absent from its fold's training rows.  Each
    benign archive is deterministically assigned to exactly one test fold, so
    aggregate predictions are out-of-fold for every eligible sample.
    """

    frame = dataset.frame if isinstance(dataset, PreparedDataset) else dataset
    required = {"filename", "label", "attack_variant"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetValidationError(
            "prepared dataset is missing split columns: " + ", ".join(missing)
        )

    positive = frame["label"].eq(1)
    variants = sorted(frame.loc[positive, "attack_variant"].astype(str).unique())
    if len(variants) < 2:
        raise DatasetValidationError("grouped evaluation requires at least two variants")
    benign_names = frame.loc[~positive, "filename"].astype(str).tolist()
    if len(benign_names) < len(variants):
        raise DatasetValidationError(
            "grouped evaluation requires at least one benign row per variant fold"
        )
    assignments = _stable_benign_assignments(
        benign_names, variants, random_state
    )
    benign_fold = frame["filename"].astype(str).map(assignments)

    all_test_indices: list[int] = []
    for variant in variants:
        test_mask = (
            (positive & frame["attack_variant"].eq(variant))
            | (~positive & benign_fold.eq(variant))
        )
        train_mask = ~test_mask
        train_variants = set(
            frame.loc[train_mask & positive, "attack_variant"].astype(str)
        )
        if variant in train_variants:
            raise AssertionError(f"variant leakage detected for {variant}")
        train_indices = frame.index[train_mask].to_numpy()
        test_indices = frame.index[test_mask].to_numpy()
        if not set(frame.loc[test_indices, "label"].astype(int)) == {0, 1}:
            raise DatasetValidationError(
                f"fold {variant!r} does not contain both classes"
            )
        all_test_indices.extend(test_indices.tolist())
        yield VariantSplit(variant, train_indices, test_indices)

    if sorted(all_test_indices) != sorted(frame.index.tolist()):
        raise AssertionError("each eligible sample must appear in exactly one test fold")


def _build_model(random_state: int) -> LGBMClassifier:
    """Construct a deterministic, CPU-portable LightGBM classifier."""

    return LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=0.1,
        random_state=random_state,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def _positive_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(features), dtype=float)
        if scores.ndim != 2 or scores.shape[1] != 2:
            raise ValueError("classifier predict_proba() must return two columns")
        result = scores[:, 1]
    elif isinstance(model, lgb.Booster) or hasattr(model, "predict"):
        result = np.asarray(model.predict(features), dtype=float).reshape(-1)
    else:
        raise TypeError("model must be a fitted LightGBM classifier or native Booster")
    if len(result) != len(features) or not np.isfinite(result).all():
        raise ValueError("model returned invalid scores")
    return np.clip(result, 0.0, 1.0)


def _compute_metrics(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    negatives = tn + fp
    positives = tp + fn
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, score))
        if len(np.unique(truth)) == 2
        else 0.0,
        "false_positive_rate": float(fp / negatives) if negatives else 0.0,
        "false_negative_rate": float(fn / positives) if positives else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _metric_confidence_intervals(
    metrics: Mapping[str, float | int]
) -> dict[str, list[float]]:
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])
    tp = int(metrics["tp"])
    return {
        "accuracy": _wilson_interval(tp + tn, tp + tn + fp + fn),
        "precision": _wilson_interval(tp, tp + fp),
        "recall": _wilson_interval(tp, tp + fn),
        "false_positive_rate": _wilson_interval(fp, fp + tn),
    }


def calibration_policy() -> dict[str, Any]:
    """Return the immutable release decision-threshold policy."""

    return {
        "id": CALIBRATION_POLICY_ID,
        "method": "next_float_above_higher_empirical_benign_quantile",
        "benign_false_positive_rate_budget": BENIGN_FPR_BUDGET,
        "benign_score_quantile": CALIBRATION_QUANTILE,
        "score_semantics": "uncalibrated_lightgbm_score",
    }


def _threshold_from_benign_scores(
    scores: Sequence[float] | pd.Series,
    labels: Sequence[int] | pd.Series,
) -> tuple[float, dict[str, Any]]:
    """Derive a deterministic threshold from benign calibration scores only."""

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if len(score_array) != len(label_array) or not len(score_array):
        raise DatasetValidationError("calibration scores and labels must align")
    if not np.isfinite(score_array).all():
        raise DatasetValidationError("calibration scores must be finite")
    benign_scores = score_array[label_array == 0]
    if not len(benign_scores):
        raise DatasetValidationError("calibration requires benign rows")
    quantile_score = float(
        np.quantile(benign_scores, CALIBRATION_QUANTILE, method="higher")
    )
    if quantile_score >= 1.0:
        raise DatasetValidationError(
            "benign calibration scores cannot satisfy the declared FPR budget"
        )
    threshold = float(np.nextafter(quantile_score, 1.0))
    exceedances = int((benign_scores >= threshold).sum())
    achieved_fpr = float(exceedances / len(benign_scores))
    return threshold, {
        **calibration_policy(),
        "threshold": threshold,
        "benign_rows": int(len(benign_scores)),
        "benign_exceedances": exceedances,
        "achieved_benign_false_positive_rate": achieved_fpr,
    }


def _grouped_oof_scores(
    frame: pd.DataFrame,
    *,
    split_random_state: int,
    model_seed_base: int,
) -> tuple[pd.Series, int]:
    """Create complete grouped OOF scores for a supplied training frame."""

    scores = pd.Series(index=frame.index, dtype=float)
    fold_count = 0
    for fold_count, split in enumerate(
        iter_variant_holdout_splits(frame, split_random_state), start=1
    ):
        inner_train = frame.loc[split.train_indices]
        inner_test = frame.loc[split.test_indices]
        model = _build_model(model_seed_base + fold_count)
        model.fit(inner_train[FEATURE_COLS], inner_train["label"].astype(int))
        scores.loc[split.test_indices] = _positive_scores(
            model, inner_test[FEATURE_COLS]
        )
    if scores.isna().any() or len(scores) != len(frame):
        raise AssertionError("grouped calibration OOF coverage is incomplete")
    return scores, fold_count


def _calibrate_grouped_oof_threshold(
    training_frame: pd.DataFrame,
    *,
    split_random_state: int,
    model_seed_base: int,
) -> tuple[float, dict[str, Any]]:
    """Calibrate solely from grouped OOF scores of the supplied training rows."""

    scores, fold_count = _grouped_oof_scores(
        training_frame,
        split_random_state=split_random_state,
        model_seed_base=model_seed_base,
    )
    threshold, details = _threshold_from_benign_scores(
        scores, training_frame["label"]
    )
    variants = sorted(
        training_frame.loc[
            training_frame["label"].eq(1), "attack_variant"
        ].astype(str).unique()
    )
    return threshold, {
        **details,
        "source_protocol": "grouped_inner_oof_on_outer_training_rows",
        "inner_rows": int(len(training_frame)),
        "inner_oof_rows": int(scores.notna().sum()),
        "inner_positive_rows": int(training_frame["label"].eq(1).sum()),
        "inner_fold_count": fold_count,
        "inner_attack_variants": variants,
        "inner_coverage_complete": bool(scores.notna().all()),
        "outer_test_rows_used": 0,
    }


def train_and_evaluate(
    features_path: str | Path = FEATURES_PATH,
    labels_path: str | Path = LABELS_PATH,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Nested grouped evaluation plus full-training threshold calibration.

    The returned aggregate metrics are strictly synthetic-development metrics.
    They must not be presented as real-world malware or zero-day performance.
    Each outer threshold is derived only from grouped OOF scores on that outer
    fold's training rows.  The release threshold is independently derived from
    complete grouped OOF scores over the full model-development dataset.
    """

    config = config or TrainingConfig()
    prepared = load_training_dataset(features_path, labels_path)
    frame = prepared.frame
    out_of_fold = pd.DataFrame(
        index=frame.index,
        columns=["prediction", "score", "threshold"],
        dtype=float,
    )
    folds: list[dict[str, Any]] = []

    for fold_number, split in enumerate(
        iter_variant_holdout_splits(prepared, config.random_state), start=1
    ):
        train = frame.loc[split.train_indices]
        test = frame.loc[split.test_indices]
        threshold, calibration = _calibrate_grouped_oof_threshold(
            train,
            split_random_state=config.random_state * 100 + fold_number,
            model_seed_base=config.random_state * 100 + fold_number * 100,
        )
        model = _build_model(config.random_state + fold_number)
        model.fit(train[FEATURE_COLS], train["label"].astype(int))
        score = _positive_scores(model, test[FEATURE_COLS])
        prediction = (score >= threshold).astype(int)
        out_of_fold.loc[split.test_indices, "prediction"] = prediction
        out_of_fold.loc[split.test_indices, "score"] = score
        out_of_fold.loc[split.test_indices, "threshold"] = threshold
        fold_metrics = _compute_metrics(test["label"], prediction, score)
        train_variants = sorted(
            train.loc[train["label"].eq(1), "attack_variant"].astype(str).unique()
        )
        test_variants = sorted(
            test.loc[test["label"].eq(1), "attack_variant"].astype(str).unique()
        )
        overlap = sorted(set(train_variants).intersection(test_variants))
        if overlap:
            raise AssertionError(
                "generated attack variant leakage: " + ", ".join(overlap)
            )
        folds.append(
            {
                "fold": fold_number,
                "held_out_attack_variant": split.held_out_variant,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_synthetic_positive_rows": int(train["label"].sum()),
                "train_benign_rows": int(train["label"].eq(0).sum()),
                "test_synthetic_positive_rows": int(test["label"].sum()),
                "test_benign_rows": int(test["label"].eq(0).sum()),
                "train_attack_variants": train_variants,
                "test_attack_variants": test_variants,
                "attack_variant_overlap": overlap,
                "threshold": threshold,
                "calibration": calibration,
                "metrics": fold_metrics,
            }
        )

    if out_of_fold.isna().any().any():
        raise AssertionError("out-of-fold predictions are incomplete")
    aggregate_metrics = _compute_metrics(
        frame["label"],
        out_of_fold["prediction"].astype(int),
        out_of_fold["score"].astype(float),
    )
    release_threshold, release_calibration = _threshold_from_benign_scores(
        out_of_fold["score"].astype(float), frame["label"]
    )
    release_calibration = {
        **release_calibration,
        "source_protocol": "full_dataset_grouped_oof",
        "oof_rows": int(out_of_fold["score"].notna().sum()),
        "positive_rows": int(frame["label"].eq(1).sum()),
        "fold_count": len(folds),
        "attack_variants": sorted(
            frame.loc[frame["label"].eq(1), "attack_variant"]
            .astype(str)
            .unique()
        ),
        "coverage_complete": bool(out_of_fold["score"].notna().all()),
    }

    final_model = _build_model(config.random_state)
    final_model.fit(frame[FEATURE_COLS], frame["label"].astype(int))
    benign_composition = prepared.summary["benign_feature_row_composition"]
    benign_composition_text = ", ".join(
        [
            f"{benign_composition.get('pypi_archive', 0):,} PyPI archives",
            f"{benign_composition.get('hard_negative', 0):,} hard negatives",
            f"{benign_composition.get('small_benign', 0):,} small benign fixtures",
            f"{benign_composition.get('office_sample', 0):,} office samples",
        ]
    )
    if benign_composition.get("other"):
        benign_composition_text += f", {benign_composition['other']:,} other rows"
    metrics_payload: dict[str, Any] = {
        "schema_version": 1,
        "extractor_feature_schema_version": EXTRACTOR_FEATURE_SCHEMA_VERSION,
        "target": {
            "id": TARGET_ID,
            "definition": TARGET_DEFINITION,
            "positive_class": "generated ZIP structural-evasion variants",
            "negative_class": "benign ZIP-compatible archives",
        },
        "scope": SYNTHETIC_SCOPE,
        "limitations": [
            "These metrics measure generated structural-evasion variants only.",
            "They do not measure malware detection or payload maliciousness.",
            (
                f"The {prepared.summary['benign_rows']:,} negative examples are "
                f"benign-labeled legacy feature rows ({benign_composition_text}); their "
                "raw archives are absent, so they cannot be re-extracted with "
                "the current parser."
            ),
            "No independently labeled real-world structural-evasion corpus is included.",
            (
                "Decision thresholds use a fixed development 0.5% benign-FPR budget "
                "and grouped training-only calibration; they are not probability calibration."
            ),
            (
                "Wilson intervals are row-level binomial benchmark intervals; "
                "they do not account for correlation among generated samples or sources."
            ),
        ],
        "feature_names": list(FEATURE_COLS),
        "dataset": {
            **prepared.summary,
            **prepared.source_sha256,
        },
        "evaluation": {
            "protocol": "nested_leave_one_generated_attack_variant_out",
            "description": (
                "Each generated positive variant is held out in full. Each outer "
                "threshold is calibrated only from grouped inner OOF scores on "
                "that outer fold's training rows."
            ),
            "random_state": config.random_state,
            "decision_policy": calibration_policy(),
            "release_threshold": release_threshold,
            "release_calibration": release_calibration,
            "leakage_check_passed": all(
                not fold["attack_variant_overlap"]
                and fold["calibration"]["outer_test_rows_used"] == 0
                and fold["calibration"]["inner_coverage_complete"]
                for fold in folds
            ),
            "out_of_fold_rows": int(len(frame)),
            "aggregate_synthetic_metrics": aggregate_metrics,
            "confidence_intervals_95_wilson": _metric_confidence_intervals(
                aggregate_metrics
            ),
            "confidence_interval_unit": "benchmark_row",
            "folds": folds,
        },
    }
    return {
        "model": final_model,
        "prepared_dataset": prepared,
        "metrics": metrics_payload,
        "out_of_fold": out_of_fold,
        "release_threshold": release_threshold,
        "calibration_policy": calibration_policy(),
    }


def train_with_cross_validation(
    features_path: str | Path = FEATURES_PATH,
    labels_path: str | Path = LABELS_PATH,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Compatibility alias for the leakage-resistant grouped evaluation."""

    warnings.warn(
        "train_with_cross_validation() now performs leave-one-attack-variant-out "
        "evaluation; use train_and_evaluate() for the explicit API",
        DeprecationWarning,
        stacklevel=2,
    )
    return train_and_evaluate(features_path, labels_path, config)


def _as_booster(model: LGBMClassifier | lgb.Booster) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    if isinstance(model, LGBMClassifier):
        if not hasattr(model, "booster_"):
            raise ValueError("LightGBM classifier must be fitted before saving")
        return model.booster_
    raise TypeError("only fitted LightGBM classifiers or native Boosters can be saved")


def save_model(
    model: LGBMClassifier | lgb.Booster,
    path: str | Path = MODEL_PATH,
    *,
    metadata: Mapping[str, Any] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Save a non-executable LightGBM text model and checksummed JSON sidecar."""

    model_path = Path(path)
    if model_path.suffix.lower() != ".txt":
        raise ValueError("native LightGBM production models must use a .txt suffix")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    booster = _as_booster(model)
    feature_names = booster.feature_name()
    if feature_names != FEATURE_COLS:
        raise ValueError(
            "model feature schema does not match ZombieGuard FEATURE_COLS"
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    booster.save_model(str(temporary))
    temporary.replace(model_path)
    model_digest = sha256_file(model_path)

    supplied = dict(metadata or {})
    protected = {
        "schema_version",
        "artifact_format",
        "model_sha256",
        "feature_names",
        "threshold",
        "extractor_feature_schema_version",
    }
    conflict = sorted(protected.intersection(supplied))
    if conflict:
        raise ValueError(
            "metadata cannot override protected fields: " + ", ".join(conflict)
        )
    sidecar: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_format": ARTIFACT_FORMAT,
        "model_sha256": model_digest,
        "feature_names": list(FEATURE_COLS),
        "threshold": threshold,
        "extractor_feature_schema_version": EXTRACTOR_FEATURE_SCHEMA_VERSION,
        "target": {"id": TARGET_ID, "definition": TARGET_DEFINITION},
        "library": {"name": "lightgbm", "version": lgb.__version__},
        **supplied,
    }
    metadata_path = metadata_path_for(model_path)
    _atomic_json_write(metadata_path, sidecar)
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_sha256": model_digest,
        "metadata_sha256": sha256_file(metadata_path),
        "metadata": sidecar,
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ModelIntegrityError(f"model metadata sidecar is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"model metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelIntegrityError("model metadata root must be a JSON object")
    return payload


def load_model(path: str | Path = MODEL_PATH) -> lgb.Booster:
    """Verify and load a native LightGBM text model without pickle execution."""

    model_path = Path(path)
    if model_path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
        raise ModelIntegrityError(
            "pickle/joblib model loading is disabled because those formats can "
            "execute code; retrain to the native .txt artifact"
        )
    if model_path.suffix.lower() != ".txt":
        raise ModelIntegrityError("expected a native LightGBM .txt model")
    if not model_path.is_file():
        raise FileNotFoundError(f"no model found at: {model_path}")

    metadata = _load_metadata(metadata_path_for(model_path))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ModelIntegrityError("unsupported model metadata schema version")
    if metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise ModelIntegrityError("unexpected model artifact format")
    if metadata.get("feature_names") != FEATURE_COLS:
        raise ModelIntegrityError("model metadata feature schema does not match code")
    if (
        metadata.get("extractor_feature_schema_version")
        != EXTRACTOR_FEATURE_SCHEMA_VERSION
    ):
        raise ModelIntegrityError(
            "model extractor/feature schema version does not match code"
        )
    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) < 1.0
    ):
        raise ModelIntegrityError("model metadata contains an invalid threshold")
    policy = metadata.get("calibration_policy")
    if policy is not None and policy != calibration_policy():
        raise ModelIntegrityError("model calibration policy does not match code")
    expected_digest = metadata.get("model_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ModelIntegrityError("model metadata has no valid SHA-256 digest")
    actual_digest = sha256_file(model_path)
    if not hmac.compare_digest(expected_digest, actual_digest):
        raise ModelIntegrityError("model checksum does not match its metadata")

    try:
        booster = lgb.Booster(model_file=str(model_path))
    except lgb.basic.LightGBMError as exc:
        raise ModelIntegrityError("native LightGBM model could not be parsed") from exc
    if booster.feature_name() != FEATURE_COLS:
        raise ModelIntegrityError("native model feature schema does not match code")
    booster._zombieguard_threshold = float(threshold)
    return booster


def _prediction_row(features: Mapping[str, Any]) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLS if column not in features]
    if missing:
        raise ValueError(
            "prediction requires every model feature; missing: " + ", ".join(missing)
        )
    row: dict[str, float] = {}
    for column in FEATURE_COLS:
        value = features[column]
        if isinstance(value, (bool, np.bool_)):
            numeric = float(int(value))
        elif isinstance(value, Real):
            numeric = float(value)
        else:
            raise TypeError(f"feature {column!r} must be numeric, got {type(value).__name__}")
        if not math.isfinite(numeric):
            raise ValueError(f"feature {column!r} must be finite")
        row[column] = numeric
    return pd.DataFrame([row], columns=FEATURE_COLS)


def _reason_codes(features: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bool(features["method_mismatch"]):
        reasons.append("HEADER_METHOD_MISMATCH")
    if bool(features["any_crc_mismatch"]):
        reasons.append("CRC_MISMATCH")
    if bool(features["lf_unknown_method"]):
        reasons.append("UNKNOWN_LOCAL_COMPRESSION_METHOD")
    if int(features["eocd_count"]) != 1:
        reasons.append("UNEXPECTED_EOCD_COUNT")
    if bool(features["declared_vs_entropy_flag"]):
        reasons.append("DECLARED_METHOD_ENTROPY_INCONSISTENCY")
    if not reasons and int(features["suspicious_entry_count"]) > 0:
        reasons.append("STRUCTURAL_INCONSISTENCY")
    return reasons


def predict(
    model: LGBMClassifier | lgb.Booster | Any,
    features: Mapping[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict structural evasion from a complete, finite feature mapping.

    No entropy-only hard override is applied: high-entropy data stored with ZIP
    method 0 is valid and must not automatically become a 100% positive verdict.
    """

    if threshold is None:
        threshold = float(
            getattr(model, "_zombieguard_threshold", DEFAULT_THRESHOLD)
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    row = _prediction_row(features)
    score = float(_positive_scores(model, row)[0])
    label = int(score >= threshold)
    return {
        "label": label,
        "verdict": "ZOMBIE ZIP DETECTED" if label else "CLEAN",
        "score": round(score, 12),
        "threshold": threshold,
        "target": TARGET_ID,
        "reason_codes": _reason_codes(features),
    }


if __name__ == "__main__":
    raise SystemExit(
        "Use `python -m zombieguard.evaluate --train --check` to train and verify artifacts."
    )
