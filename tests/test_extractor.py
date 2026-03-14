import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entropy import compute_shannon_entropy, compute_renyi_entropy
from src.extractor import extract_features


def test_compressed_data_high_entropy():
	# DEFLATE compressed bytes are chaotic - entropy must be above 7.0
	import os
	import zlib

	compressed = zlib.compress(os.urandom(32768))
	entropy = compute_shannon_entropy(compressed)
	assert entropy > 7.0, f"Expected > 7.0, got {entropy}"


def test_plain_text_low_entropy():
	# Plain readable text has patterns - entropy must be below 5.0
	plain = b"hello world " * 1000
	entropy = compute_shannon_entropy(plain)
	assert entropy < 5.0, f"Expected < 5.0, got {entropy}"


def test_normal_zip_no_flags():
	features = extract_features("tests/fixtures/normal.zip")
	assert features["method_mismatch"] == False
	assert features["declared_vs_entropy_flag"] == False
	assert features["lf_compression_method"] == 8


def test_zombie_zip_flagged():
	features = extract_features("tests/fixtures/zombie.zip")
	# Header lies - claims STORE but data is compressed
	assert features["declared_vs_entropy_flag"] == True
	# LFH says STORE (0), CDH says DEFLATE (8) - mismatch
	assert features["method_mismatch"] == True
	assert features["lf_compression_method"] == 0
	assert features["cd_compression_method"] == 8


def test_eocd_count_normal():
	features = extract_features("tests/fixtures/normal.zip")
	assert features["eocd_count"] == 1


def test_feature_keys_present():
	features = extract_features("tests/fixtures/normal.zip")
	expected_keys = [
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
	for key in expected_keys:
		assert key in features, f"Missing feature: {key}"
