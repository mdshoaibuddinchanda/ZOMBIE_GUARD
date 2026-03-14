"""
extractor.py
Feature extraction for ZIP structural and entropy-based evasion signals.
Part of ZombieGuard - Archive Header Evasion Detection System.
CVE-2026-0866 | https://github.com/YOUR_USERNAME/zombieguard
"""

import os
import struct

from src.entropy import compute_renyi_entropy, compute_shannon_entropy

# ZIP format constants
LFH_SIGNATURE = b"PK\x03\x04"  # Local File Header signature
CDH_SIGNATURE = b"PK\x01\x02"  # Central Directory Header signature
EOCD_SIGNATURE = b"PK\x05\x06"  # End of Central Directory signature

# Compression method codes
METHOD_STORE = 0
ENTROPY_COMPRESSED_THRESHOLD = 7.0  # bytes above this = behaving like compressed data
KNOWN_METHODS = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 19, 97, 98}


def _find_all_signatures(data: bytes, signature: bytes) -> list:
	"""Find all positions of a signature in raw bytes."""
	positions = []
	start = 0
	while True:
		pos = data.find(signature, start)
		if pos == -1:
			break
		positions.append(pos)
		start = pos + 1
	return positions


def _parse_local_file_header(data: bytes, offset: int) -> dict:
	"""
	Parse Local File Header starting at given offset.
	LFH structure (from ZIP spec section 4.3.4):
	  Offset  Length  Field
	  0       4       Signature (50 4B 03 04)
	  4       2       Version needed
	  6       2       General purpose bit flag
	  8       2       Compression method   <-- THIS IS THE KEY FIELD
	  10      2       Last mod time
	  12      2       Last mod date
	  14      4       CRC-32
	  18      4       Compressed size
	  22      4       Uncompressed size
	  26      2       Filename length
	  28      2       Extra field length
	  30+     n       Filename
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
	return {
		"compression_method": method,
		"crc32": crc32,
		"compressed_size": comp_size,
		"uncompressed_size": uncomp_size,
		"filename": filename,
		"data_offset": data_offset,
		"fname_len": fname_len,
	}


def _parse_central_directory_header(data: bytes, offset: int) -> dict:
	"""
	Parse Central Directory Header starting at given offset.
	CDH structure (from ZIP spec section 4.3.12):
	  Offset  Length  Field
	  0       4       Signature (50 4B 01 02)
	  4       2       Version made by
	  6       2       Version needed
	  8       2       General purpose bit flag
	  10      2       Compression method   <-- compare with LFH method
	  12      2       Last mod time
	  14      2       Last mod date
	  16      4       CRC-32
	  20      4       Compressed size
	  24      4       Uncompressed size
	  28      2       Filename length
	  30      2       Extra field length
	  32      2       File comment length
	  34      2       Disk number start
	  36      2       Internal attributes
	  38      4       External attributes
	  42      4       Relative offset of local header
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


def extract_features(zip_filepath: str) -> dict:
	"""
	Extract all 9 detection features from a ZIP file.
	Returns a dictionary with these keys:
	  - lf_compression_method     : int   (method declared in Local File Header)
	  - cd_compression_method     : int   (method declared in Central Directory)
	  - method_mismatch           : bool  (LFH method != CDH method)
	  - data_entropy_shannon      : float (Shannon entropy of actual data bytes)
	  - data_entropy_renyi        : float (Renyi entropy alpha=2 of actual data bytes)
	  - declared_vs_entropy_flag  : bool  (claims STORE but entropy says compressed)
	  - eocd_count                : int   (number of EOCD signatures - >1 = Gootloader)
	  - file_size_bytes           : int   (total file size)
	  - lf_unknown_method         : int   (1 when LFH method code is outside known ZIP methods)
	"""
	features = {
		"lf_compression_method": -1,
		"cd_compression_method": -1,
		"method_mismatch": False,
		"data_entropy_shannon": 0.0,
		"data_entropy_renyi": 0.0,
		"declared_vs_entropy_flag": False,
		"eocd_count": 0,
		"file_size_bytes": 0,
		"lf_unknown_method": 0,
	}

	if not os.path.isfile(zip_filepath):
		return features

	with open(zip_filepath, "rb") as file:
		data = file.read()

	features["file_size_bytes"] = len(data)

	# -- Feature: EOCD count (Gootloader uses 500-1000 concatenated ZIPs)
	eocd_positions = _find_all_signatures(data, EOCD_SIGNATURE)
	features["eocd_count"] = len(eocd_positions)

	# -- Parse first Local File Header
	lfh_positions = _find_all_signatures(data, LFH_SIGNATURE)
	if not lfh_positions:
		return features

	lfh = _parse_local_file_header(data, lfh_positions[0])
	if lfh:
		features["lf_compression_method"] = lfh["compression_method"]

		# -- Feature: entropy of actual data bytes
		data_start = lfh["data_offset"]
		data_end = data_start + lfh["compressed_size"]
		payload_bytes = data[data_start:data_end]

		if len(payload_bytes) > 0:
			shannon = compute_shannon_entropy(payload_bytes)
			renyi = compute_renyi_entropy(payload_bytes, alpha=2.0)
			features["data_entropy_shannon"] = round(shannon, 4)
			features["data_entropy_renyi"] = round(renyi, 4)

			# -- Feature: header claims STORE but data behaves like compressed
			claims_store = lfh["compression_method"] == METHOD_STORE
			looks_compressed = shannon > ENTROPY_COMPRESSED_THRESHOLD
			features["declared_vs_entropy_flag"] = bool(claims_store and looks_compressed)

	# -- Parse first Central Directory Header
	cdh_positions = _find_all_signatures(data, CDH_SIGNATURE)
	if cdh_positions:
		cdh = _parse_central_directory_header(data, cdh_positions[0])
		if cdh:
			features["cd_compression_method"] = cdh["compression_method"]

	# -- Feature: method mismatch between LFH and CDH
	if features["lf_compression_method"] != -1 and features["cd_compression_method"] != -1:
		features["method_mismatch"] = (
			features["lf_compression_method"] != features["cd_compression_method"]
		)

	# -- Feature: unknown/undefined LFH method code in the wild
	if features["lf_compression_method"] != -1:
		features["lf_unknown_method"] = int(
			features["lf_compression_method"] not in KNOWN_METHODS
		)

	return features
