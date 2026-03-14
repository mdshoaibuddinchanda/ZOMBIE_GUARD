"""
detector.py
ZombieGuard - Command-line detector for Zombie ZIP and archive evasion.
Usage:
	python src/detector.py <zipfile>
	python src/detector.py <zipfile> --verbose
	python src/detector.py <zipfile> --threshold 0.7
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import load_model, predict
from src.extractor import extract_features

# -- Thresholds -------------------------------------------------
DEFAULT_THRESHOLD = 0.5
HIGH_CONFIDENCE = 0.85

# Format-specific detection thresholds calibrated from generalisation runs.
FORMAT_THRESHOLDS = {
	".zip": 0.50,
	".apk": 0.50,
	".jar": 0.50,
	".rar": 0.15,
	".7z": 0.25,
}

# -- Color codes for terminal output ----------------------------
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_threshold(filepath: str, default_threshold: float = DEFAULT_THRESHOLD) -> float:
	"""Return a format-aware threshold for a given file path."""
	ext = os.path.splitext(filepath)[1].lower()
	return FORMAT_THRESHOLDS.get(ext, default_threshold)


def format_verdict(result: dict, threshold: float) -> str:
	"""Format colorized detection verdict text from prediction output."""
	prob = result["probability"]
	label = 1 if prob >= threshold else 0

	if label == 1:
		if prob >= HIGH_CONFIDENCE:
			confidence = "HIGH CONFIDENCE"
			color = RED
		else:
			confidence = "LOW CONFIDENCE"
			color = YELLOW
		return (
			f"{color}{BOLD}"
			f"[ZOMBIE ZIP DETECTED] {confidence} "
			f"(probability: {prob:.1%})"
			f"{RESET}"
		)
	return (
		f"{GREEN}{BOLD}"
		f"[CLEAN] No evasion pattern detected "
		f"(probability: {prob:.1%})"
		f"{RESET}"
	)


def print_features(features: dict):
	"""Print extracted feature values with suspicious-signal highlighting."""
	print(f"\n{BLUE}-- Extracted Features ---------------------------{RESET}")

	signals = {
		"LFH compression method": features.get("lf_compression_method", -1),
		"CDH compression method": features.get("cd_compression_method", -1),
		"Method mismatch (LFH!=CDH)": features.get("method_mismatch", False),
		"Shannon entropy": f"{features.get('data_entropy_shannon', 0):.4f}",
		"Renyi entropy": f"{features.get('data_entropy_renyi', 0):.4f}",
		"Declared-vs-entropy flag": features.get("declared_vs_entropy_flag", False),
		"EOCD count": features.get("eocd_count", 0),
		"File size": f"{features.get('file_size_bytes', 0):,} bytes",
	}

	for name, value in signals.items():
		# Highlight suspicious values
		is_suspicious = (
			(name == "Method mismatch (LFH!=CDH)" and value is True)
			or (name == "Declared-vs-entropy flag" and value is True)
			or (name == "EOCD count" and isinstance(value, int) and value > 1)
		)
		marker = f"{RED}  <- SUSPICIOUS{RESET}" if is_suspicious else ""
		print(f"  {name:<35} {value}{marker}")

	print(f"{BLUE}-----------------------------------------------{RESET}")


def detect_file(
	filepath: str,
	verbose: bool = False,
	threshold: float = DEFAULT_THRESHOLD,
	model_path: str = "models/xgboost_model.pkl",
) -> dict:
	"""
	Run full detection on a single ZIP file.
	Returns the result dictionary.
	"""
	if not os.path.isfile(filepath):
		print(f"{RED}ERROR: File not found: {filepath}{RESET}")
		sys.exit(1)

	if not filepath.lower().endswith(".zip"):
		print(
			f"{YELLOW}WARNING: File does not have .zip extension. "
			f"Analyzing anyway...{RESET}"
		)

	# Load model
	try:
		model = load_model(model_path)
	except FileNotFoundError:
		print(f"{RED}ERROR: No trained model found at {model_path}{RESET}")
		print("Run: python src/classifier.py")
		sys.exit(1)

	# Extract features
	features = extract_features(filepath)

	# Predict
	result = predict(model, features)
	effective_threshold = threshold
	if threshold == DEFAULT_THRESHOLD:
		effective_threshold = get_threshold(filepath, DEFAULT_THRESHOLD)
	prob = result["probability"]
	label = 1 if prob >= effective_threshold else 0
	result["label"] = label
	result["verdict"] = "ZOMBIE ZIP DETECTED" if label == 1 else "CLEAN"
	result["threshold"] = effective_threshold

	# Print output
	filename = os.path.basename(filepath)
	print(f"\n{BOLD}ZombieGuard - Archive Evasion Detector{RESET}")
	print(f"{'-' * 45}")
	print(f"File    : {filename}")
	print(f"Size    : {features.get('file_size_bytes', 0):,} bytes")
	print(f"Threshold: {effective_threshold:.2f}")
	print(f"Verdict : {format_verdict(result, effective_threshold)}")

	if verbose:
		print_features(features)

	print()
	return result


def batch_detect(
	directory: str,
	verbose: bool = False,
	threshold: float = DEFAULT_THRESHOLD,
	model_path: str = "models/xgboost_model.pkl",
):
	"""Scan all ZIP files in a directory and print aggregate counts."""
	zip_files = [f for f in os.listdir(directory) if f.lower().endswith(".zip")]

	if not zip_files:
		print(f"No ZIP files found in: {directory}")
		return

	print(f"\n{BOLD}ZombieGuard - Batch Scan{RESET}")
	print(f"Scanning {len(zip_files)} files in: {directory}\n")

	detected = 0
	clean = 0

	for fname in sorted(zip_files):
		fpath = os.path.join(directory, fname)
		model = load_model(model_path)
		features = extract_features(fpath)
		result = predict(model, features)
		prob = result["probability"]
		effective_threshold = threshold
		if threshold == DEFAULT_THRESHOLD:
			effective_threshold = get_threshold(fpath, DEFAULT_THRESHOLD)
		label = 1 if prob >= effective_threshold else 0

		if label == 1:
			detected += 1
			print(f"  {RED}[DETECTED]{RESET} {fname:<50} ({prob:.1%})")
		else:
			clean += 1
			print(f"  {GREEN}[CLEAN]   {RESET} {fname:<50} ({prob:.1%})")

	print(f"\n{'-' * 55}")
	print(f"Scanned : {len(zip_files)} files")
	print(f"{RED}Detected: {detected} Zombie ZIP(s){RESET}")
	print(f"{GREEN}Clean   : {clean} file(s){RESET}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="ZombieGuard - Detect Zombie ZIP archive evasion",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python src/detector.py suspicious.zip
  python src/detector.py suspicious.zip --verbose
  python src/detector.py suspicious.zip --threshold 0.7
  python src/detector.py --batch data/raw/malicious/
		""",
	)

	parser.add_argument("file", nargs="?", help="Path to ZIP file to analyze")
	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Show all extracted feature values",
	)
	parser.add_argument(
		"--threshold",
		"-t",
		type=float,
		default=DEFAULT_THRESHOLD,
		help=f"Detection threshold (default: {DEFAULT_THRESHOLD})",
	)
	parser.add_argument(
		"--batch",
		"-b",
		metavar="DIRECTORY",
		help="Scan all ZIP files in a directory",
	)
	parser.add_argument(
		"--model",
		default="models/xgboost_model.pkl",
		help="Path to trained model file",
	)

	args = parser.parse_args()

	if args.batch:
		batch_detect(args.batch, args.verbose, args.threshold, args.model)
	elif args.file:
		result = detect_file(args.file, args.verbose, args.threshold, args.model)
		sys.exit(0 if result["label"] == 0 else 1)
	else:
		parser.print_help()
