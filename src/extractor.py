"""
extractor.py
ZIP structural and entropy feature extractor for ZombieGuard.
Extracts 15 features per archive covering all entries, not just
the first. Includes CRC32 verification and encryption detection.
Part of ZombieGuard - Archive Header Evasion Detection System.
CVE-2026-0866 | https://github.com/mdshoaibuddinchanda/zombieguard
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Dict, List

from src.entropy import compute_renyi_entropy, compute_shannon_entropy

# -- ZIP format signatures ----------------------------------------------------
LFH_SIGNATURE = b"PK\x03\x04"
CDH_SIGNATURE = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"

# -- Compression method constants --------------------------------------------
METHOD_STORE = 0
METHOD_DEFLATE = 8

# -- Known valid ZIP compression method codes --------------------------------
# Per PKWARE APPNOTE.TXT section 4.4.5
KNOWN_METHODS = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 19, 97, 98}

# -- Detection threshold ------------------------------------------------------
ENTROPY_COMPRESSED_THRESHOLD = 7.0


# -- Low-level parsers --------------------------------------------------------
def _find_all_signatures(data: bytes, signature: bytes) -> List[int]:
	"""Return byte offsets of every occurrence of signature in data."""
	positions = []
	start = 0
	while True:
		pos = data.find(signature, start)
		if pos == -1:
			break
		positions.append(pos)
		start = pos + 1
	return positions


def _parse_lfh(data: bytes, offset: int) -> Dict | None:
	"""
	Parse one Local File Header starting at offset.

	ZIP LFH layout (APPNOTE.TXT 4.3.7):
	  Offset  Len  Field
	  0       4    Signature (50 4B 03 04)
	  4       2    Version needed
	  6       2    General purpose bit flag
	  8       2    Compression method
	  10      2    Last mod time
	  12      2    Last mod date
	  14      4    CRC-32
	  18      4    Compressed size
	  22      4    Uncompressed size
	  26      2    Filename length
	  28      2    Extra field length
	  30+     n    Filename
	"""
	if offset + 30 > len(data):
		return None

	fields = struct.unpack_from("<4sHHHHHIIIHH", data, offset)
	(
		sig,
		ver,
		flags,
		method,
		mod_time,
		mod_date,
		crc32,
		comp_size,
		uncomp_size,
		fname_len,
		extra_len,
	) = fields

	if sig != LFH_SIGNATURE:
		return None

	fname_start = offset + 30
	fname_end = fname_start + fname_len
	filename = data[fname_start:fname_end].decode("utf-8", errors="replace")
	data_offset = fname_end + extra_len

	# Bit 0 of general purpose bit flag = encryption
	is_encrypted = bool(flags & 0x0001)

	return {
		"compression_method": method,
		"crc32": crc32,
		"compressed_size": comp_size,
		"uncompressed_size": uncomp_size,
		"filename": filename,
		"data_offset": data_offset,
		"is_encrypted": is_encrypted,
	}


def _parse_cdh(data: bytes, offset: int) -> Dict | None:
	"""
	Parse one Central Directory Header starting at offset.

	ZIP CDH layout (APPNOTE.TXT 4.3.12):
	  Offset  Len  Field
	  0       4    Signature (50 4B 01 02)
	  4       2    Version made by
	  6       2    Version needed
	  8       2    General purpose bit flag
	  10      2    Compression method
	  12      2    Last mod time
	  14      2    Last mod date
	  16      4    CRC-32
	  20      4    Compressed size
	  24      4    Uncompressed size
	  28      2    Filename length
	  30      2    Extra field length
	  32      2    File comment length
	  34      2    Disk number start
	  36      2    Internal attributes
	  38      4    External attributes
	  42      4    Relative offset of local header
	"""
	if offset + 46 > len(data):
		return None

	fields = struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, offset)
	sig = fields[0]
	method = fields[4]
	crc32 = fields[7]
	fname_len = fields[10]

	if sig != CDH_SIGNATURE:
		return None

	fname_start = offset + 46
	filename = data[fname_start : fname_start + fname_len].decode("utf-8", errors="replace")

	return {
		"compression_method": method,
		"crc32": crc32,
		"filename": filename,
	}


# -- Per-entry analysis -------------------------------------------------------
def _analyse_all_entries(data: bytes) -> Dict:
	"""
	Parse every LFH entry in the archive and return
	aggregate statistics across all entries.

	This is the core improvement over single-entry analysis.
	Multi-file ZIPs hide malicious entries beyond entry 1.
	"""
	lfh_positions = _find_all_signatures(data, LFH_SIGNATURE)
	cdh_positions = _find_all_signatures(data, CDH_SIGNATURE)

	if not lfh_positions:
		return {
			"first_lfh": None,
			"first_cdh": None,
			"entry_count": 0,
			"suspicious_entry_count": 0,
			"suspicious_entry_ratio": 0.0,
			"max_entropy_shannon": 0.0,
			"max_entropy_renyi": 0.0,
			"entropy_variance": 0.0,
			"any_encrypted": False,
			"any_unknown_method": False,
			"any_crc_mismatch": False,
			"lf_crc_valid": True,
		}

	entry_entropies = []
	suspicious_count = 0
	any_encrypted = False
	any_unknown_method = False
	any_crc_mismatch = False
	first_lfh_data = None
	first_crc_valid = True

	# Build CDH lookup by filename for fast comparison
	cdh_by_fname: Dict[str, Dict] = {}
	for pos in cdh_positions:
		cdh = _parse_cdh(data, pos)
		if cdh:
			cdh_by_fname[cdh["filename"]] = cdh

	for i, pos in enumerate(lfh_positions):
		lfh = _parse_lfh(data, pos)
		if not lfh:
			continue

		if i == 0:
			first_lfh_data = lfh

		# -- Encryption check
		if lfh["is_encrypted"]:
			any_encrypted = True

		# -- Unknown method check
		if lfh["compression_method"] not in KNOWN_METHODS and lfh["compression_method"] != -1:
			any_unknown_method = True

		# -- Payload entropy
		data_start = lfh["data_offset"]
		data_end = data_start + max(lfh["compressed_size"], 1)
		payload = data[data_start:data_end]

		if len(payload) > 0:
			shannon = compute_shannon_entropy(payload)
			entry_entropies.append(shannon)

			# -- CRC32 verification
			if not lfh["is_encrypted"] and len(payload) > 0:
				try:
					# Try to verify CRC - only works if data is stored
					if lfh["compression_method"] == METHOD_STORE:
						actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
						if actual_crc != lfh["crc32"]:
							any_crc_mismatch = True
							if i == 0:
								first_crc_valid = False
				except Exception:
					pass

		# -- Method mismatch check vs CDH
		cdh = cdh_by_fname.get(lfh["filename"])
		if cdh:
			if lfh["compression_method"] != cdh["compression_method"]:
				suspicious_count += 1
			# CRC mismatch between LFH and CDH
			if lfh["crc32"] != cdh["crc32"]:
				any_crc_mismatch = True
		else:
			# No matching CDH entry - suspicious
			suspicious_count += 1

	# -- Aggregate stats
	import numpy as np

	entry_count = len(lfh_positions)

	return {
		"first_lfh": first_lfh_data,
		"first_cdh": cdh_by_fname.get(first_lfh_data["filename"]) if first_lfh_data else None,
		"entry_count": entry_count,
		"suspicious_entry_count": suspicious_count,
		"suspicious_entry_ratio": round(suspicious_count / entry_count, 4) if entry_count > 0 else 0.0,
		"max_entropy_shannon": round(float(np.max(entry_entropies)), 4) if entry_entropies else 0.0,
		"max_entropy_renyi": 0.0,
		"entropy_variance": round(float(np.var(entry_entropies)), 4) if len(entry_entropies) > 1 else 0.0,
		"any_encrypted": any_encrypted,
		"any_unknown_method": any_unknown_method,
		"any_crc_mismatch": any_crc_mismatch,
		"lf_crc_valid": first_crc_valid,
	}


# -- Main feature extractor ---------------------------------------------------
def extract_features(zip_filepath: str) -> Dict:
	"""
	Extract all 15 detection features from a ZIP file.

	Features cover:
	  - Per-entry structural inconsistencies (all entries)
	  - Payload entropy signals (max across all entries)
	  - CRC32 verification
	  - Encryption flag detection
	  - EOCD signature count (concatenation detection)

	Returns a dictionary with 15 feature keys.
	Returns default values (zeros/False) if file cannot be read.
	"""
	defaults = {
		"lf_compression_method": -1,
		"cd_compression_method": -1,
		"method_mismatch": False,
		"data_entropy_shannon": 0.0,
		"data_entropy_renyi": 0.0,
		"declared_vs_entropy_flag": False,
		"eocd_count": 0,
		"lf_unknown_method": False,
		# NEW per-entry features
		"entry_count": 0,
		"suspicious_entry_count": 0,
		"suspicious_entry_ratio": 0.0,
		"entropy_variance": 0.0,
		# NEW CRC features
		"lf_crc_valid": True,
		"any_crc_mismatch": False,
		# NEW encryption feature
		"is_encrypted": False,
		# Keep for export/analysis - not in model
		"file_size_bytes": 0,
	}

	if not os.path.isfile(zip_filepath):
		return defaults

	try:
		with open(zip_filepath, "rb") as file:
			data = file.read()
	except Exception:
		return defaults

	features = defaults.copy()
	features["file_size_bytes"] = len(data)

	# -- EOCD count (Gootloader concatenation signal)
	eocd_positions = _find_all_signatures(data, EOCD_SIGNATURE)
	features["eocd_count"] = len(eocd_positions)

	# -- Full per-entry analysis
	analysis = _analyse_all_entries(data)

	features["entry_count"] = analysis["entry_count"]
	features["suspicious_entry_count"] = analysis["suspicious_entry_count"]
	features["suspicious_entry_ratio"] = analysis["suspicious_entry_ratio"]
	features["entropy_variance"] = analysis["entropy_variance"]
	features["any_crc_mismatch"] = analysis["any_crc_mismatch"]
	features["lf_crc_valid"] = analysis["lf_crc_valid"]
	features["is_encrypted"] = analysis["any_encrypted"]
	features["lf_unknown_method"] = int(analysis["any_unknown_method"])

	# -- First entry signals (primary detection signals)
	lfh = analysis["first_lfh"]
	cdh = analysis["first_cdh"]

	if lfh:
		features["lf_compression_method"] = lfh["compression_method"]

		# Max entropy across all entries (stronger signal than first only)
		features["data_entropy_shannon"] = analysis["max_entropy_shannon"]

		# Compute Renyi entropy on first entry payload for consistency
		data_start = lfh["data_offset"]
		data_end = data_start + max(lfh["compressed_size"], 1)
		payload = data[data_start:data_end]

		if len(payload) > 0:
			renyi = compute_renyi_entropy(payload, alpha=2.0)
			features["data_entropy_renyi"] = round(renyi, 4)

		# Declared-vs-entropy mismatch
		claims_store = lfh["compression_method"] == METHOD_STORE
		looks_compressed = features["data_entropy_shannon"] > ENTROPY_COMPRESSED_THRESHOLD
		features["declared_vs_entropy_flag"] = bool(claims_store and looks_compressed)

	if cdh:
		features["cd_compression_method"] = cdh["compression_method"]

	# -- Method mismatch (first entry LFH vs CDH)
	if features["lf_compression_method"] != -1 and features["cd_compression_method"] != -1:
		features["method_mismatch"] = (
			features["lf_compression_method"] != features["cd_compression_method"]
		)

	return features
