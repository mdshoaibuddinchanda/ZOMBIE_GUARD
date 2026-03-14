"""
tests/test_classifier.py
Automated integration tests for ZombieGuard pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
import subprocess

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import extract_features
from src.classifier import load_model, predict

# -- Fixtures ---------------------------------------------------------------
ZOMBIE_ZIP = "tests/fixtures/zombie.zip"
NORMAL_ZIP = "tests/fixtures/normal.zip"
MODEL_PATH = "models/xgboost_model.pkl"
FEATURES_CSV = "data/processed/features.csv"
LABELS_CSV = "data/processed/labels.csv"


# -- Prerequisite checks ----------------------------------------------------
def test_required_files_exist():
	"""All required project files must exist before running."""
	required = [ZOMBIE_ZIP, NORMAL_ZIP, MODEL_PATH, FEATURES_CSV, LABELS_CSV]
	for path in required:
		assert os.path.isfile(path), f"Required file missing: {path}"


def test_model_loads():
	"""Model must load without error."""
	model = load_model(MODEL_PATH)
	assert model is not None


# -- Extractor tests --------------------------------------------------------
def test_zombie_zip_features():
	"""Zombie ZIP must produce suspicious feature values."""
	features = extract_features(ZOMBIE_ZIP)
	assert features["lf_compression_method"] == 0, "LFH must declare STORE (0)"
	assert features["cd_compression_method"] == 8, "CDH must declare DEFLATE (8)"
	assert features["method_mismatch"] is True, "Method mismatch must be True"
	assert features["data_entropy_shannon"] > 7.0, "Payload entropy must be high"
	assert features["declared_vs_entropy_flag"] is True, "Entropy flag must fire"
	assert features["eocd_count"] == 1, "Normal EOCD count"


def test_normal_zip_features():
	"""Normal ZIP must produce clean feature values."""
	features = extract_features(NORMAL_ZIP)
	assert features["lf_compression_method"] == 8, "LFH must declare DEFLATE (8)"
	assert features["cd_compression_method"] == 8, "CDH must declare DEFLATE (8)"
	assert features["method_mismatch"] is False, "No method mismatch"
	assert features["declared_vs_entropy_flag"] is False, "No entropy flag"
	assert features["eocd_count"] == 1, "Normal EOCD count"


def test_all_9_features_present():
	"""Feature extractor must return all expected feature keys."""
	features = extract_features(ZOMBIE_ZIP)
	expected = [
		"lf_compression_method",
		"cd_compression_method",
		"method_mismatch",
		"data_entropy_shannon",
		"data_entropy_renyi",
		"declared_vs_entropy_flag",
		"eocd_count",
		"file_size_bytes",
		"lf_unknown_method",
	]
	for key in expected:
		assert key in features, f"Missing feature key: {key}"


def test_missing_file_returns_defaults():
	"""Extractor must not crash on missing file - return default dict."""
	features = extract_features("nonexistent_file.zip")
	assert isinstance(features, dict)
	assert features["file_size_bytes"] == 0


# -- Classifier / predict tests ---------------------------------------------
def test_zombie_zip_classified_malicious():
	"""Zombie ZIP must be classified as malicious."""
	model = load_model(MODEL_PATH)
	features = extract_features(ZOMBIE_ZIP)
	result = predict(model, features)
	assert result["label"] == 1, "Must be classified malicious"
	assert result["probability"] > 0.9, "Confidence must be above 90%"
	assert result["verdict"] == "ZOMBIE ZIP DETECTED"


def test_normal_zip_classified_clean():
	"""Normal ZIP must be classified as clean."""
	model = load_model(MODEL_PATH)
	features = extract_features(NORMAL_ZIP)
	result = predict(model, features)
	assert result["label"] == 0, "Must be classified clean"
	assert result["probability"] < 0.3, "Confidence must be below 30%"
	assert result["verdict"] == "CLEAN"


def test_predict_returns_required_keys():
	"""Predict must always return label, verdict, and probability."""
	model = load_model(MODEL_PATH)
	features = extract_features(ZOMBIE_ZIP)
	result = predict(model, features)
	assert "label" in result
	assert "verdict" in result
	assert "probability" in result
	assert isinstance(result["probability"], float)
	assert 0.0 <= result["probability"] <= 1.0


# -- Dataset integrity tests -------------------------------------------------
def test_dataset_no_nan():
	"""Dataset must have zero NaN values."""
	features_df = pd.read_csv(FEATURES_CSV)
	nan_count = features_df.isna().sum().sum()
	assert nan_count == 0, f"Dataset has {nan_count} NaN values"


def test_dataset_has_both_classes():
	"""Dataset must contain both malicious and benign samples."""
	labels_df = pd.read_csv(LABELS_CSV)
	assert (labels_df["label"] == 1).sum() > 0, "No malicious samples"
	assert (labels_df["label"] == 0).sum() > 0, "No benign samples"


def test_dataset_minimum_size():
	"""Dataset must meet minimum size for credible evaluation."""
	labels_df = pd.read_csv(LABELS_CSV)
	assert len(labels_df) >= 1000, f"Dataset too small: {len(labels_df)} samples (minimum 1000)"


def test_dataset_balance():
	"""Dataset must not be severely imbalanced (max 4:1 ratio)."""
	labels_df = pd.read_csv(LABELS_CSV)
	mal = (labels_df["label"] == 1).sum()
	ben = (labels_df["label"] == 0).sum()
	ratio = max(mal, ben) / min(mal, ben)
	assert ratio <= 4.0, f"Dataset too imbalanced: {mal} malicious vs {ben} benign (ratio {ratio:.1f})"


# -- CLI integration test ----------------------------------------------------
def test_cli_zombie_returns_exit_code_1():
	"""CLI must return exit code 1 (detected) for Zombie ZIP."""
	result = subprocess.run(
		["python", "src/detector.py", ZOMBIE_ZIP],
		capture_output=True,
		text=True,
	)
	assert result.returncode == 1, f"Expected exit code 1 (detected), got {result.returncode}"
	assert "ZOMBIE ZIP DETECTED" in result.stdout


def test_cli_normal_returns_exit_code_0():
	"""CLI must return exit code 0 (clean) for normal ZIP."""
	result = subprocess.run(
		["python", "src/detector.py", NORMAL_ZIP],
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0, f"Expected exit code 0 (clean), got {result.returncode}"
	assert "CLEAN" in result.stdout


def test_cli_verbose_shows_features():
	"""Verbose flag must show extracted feature values."""
	result = subprocess.run(
		["python", "src/detector.py", ZOMBIE_ZIP, "--verbose"],
		capture_output=True,
		text=True,
	)
	assert "Shannon entropy" in result.stdout
	assert "Method mismatch" in result.stdout
	assert "SUSPICIOUS" in result.stdout
