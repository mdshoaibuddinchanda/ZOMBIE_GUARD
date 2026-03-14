"""
entropy.py
Entropy utilities for byte-level randomness analysis in archive payloads.
Part of ZombieGuard - Archive Header Evasion Detection System.
CVE-2026-0866 | https://github.com/YOUR_USERNAME/zombieguard
"""

import numpy as np


def compute_shannon_entropy(byte_data: bytes) -> float:
	"""Compute Shannon entropy over a byte sequence."""
	if len(byte_data) == 0:
		return 0.0
	counts = np.bincount(np.frombuffer(byte_data, dtype=np.uint8), minlength=256)
	probabilities = counts / len(byte_data)
	probabilities = probabilities[probabilities > 0]
	return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_renyi_entropy(byte_data: bytes, alpha: float = 2.0) -> float:
	"""Compute Renyi entropy over a byte sequence for a given alpha."""
	if len(byte_data) == 0:
		return 0.0
	if alpha == 1.0:
		return compute_shannon_entropy(byte_data)
	counts = np.bincount(np.frombuffer(byte_data, dtype=np.uint8), minlength=256)
	probabilities = counts / len(byte_data)
	probabilities = probabilities[probabilities > 0]
	return float((1.0 / (1.0 - alpha)) * np.log2(np.sum(probabilities ** alpha)))
