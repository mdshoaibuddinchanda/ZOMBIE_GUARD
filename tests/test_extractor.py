"""Deterministic structural tests for the bounded ZIP extractor."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from zombieguard.extractor import extract_features

LFH = b"PK\x03\x04"
CDH = b"PK\x01\x02"
EOCD = b"PK\x05\x06"
DATA_DESCRIPTOR = b"PK\x07\x08"


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, wbits=-15)
    return compressor.compress(data) + compressor.flush()


def _build_zip(entries: list[dict[str, Any]]) -> bytes:
    """Build a small classic ZIP without relying on timestamps or randomness."""
    local_records = bytearray()
    central_records: list[bytes] = []

    for entry in entries:
        filename = entry["filename"]
        plain = entry["data"]
        central_method = entry.get("central_method", 8)
        local_method = entry.get("local_method", central_method)
        uses_descriptor = entry.get("data_descriptor", False)
        flags = entry.get("flags", 0x0008 if uses_descriptor else 0)
        payload = plain if central_method == 0 else _raw_deflate(plain)
        crc32 = zlib.crc32(plain) & 0xFFFFFFFF
        local_offset = len(local_records)

        local_records.extend(
            struct.pack(
                "<4sHHHHHIIIHH",
                LFH,
                20,
                flags,
                local_method,
                0,
                0,
                0 if uses_descriptor else crc32,
                0 if uses_descriptor else len(payload),
                0 if uses_descriptor else len(plain),
                len(filename),
                0,
            )
        )
        local_records.extend(filename)
        local_records.extend(payload)
        if uses_descriptor:
            local_records.extend(
                struct.pack("<4sIII", DATA_DESCRIPTOR, crc32, len(payload), len(plain))
            )

        central_records.append(
            struct.pack(
                "<4sHHHHHHIIIHHHHHII",
                CDH,
                20,
                20,
                flags,
                central_method,
                0,
                0,
                crc32,
                len(payload),
                len(plain),
                len(filename),
                0,
                0,
                0,
                0,
                0,
                local_offset,
            )
            + filename
        )

    central_offset = len(local_records)
    central_directory = b"".join(central_records)
    return (
        bytes(local_records)
        + central_directory
        + struct.pack(
            "<4sHHHHIIH",
            EOCD,
            0,
            0,
            len(entries),
            len(entries),
            len(central_directory),
            central_offset,
            0,
        )
    )


def _extract(tmp_path: Path, archive: bytes, **kwargs: Any) -> dict[str, Any]:
    archive_path = tmp_path / "sample.zip"
    archive_path.write_bytes(archive)
    return extract_features(archive_path, **kwargs)


def test_normal_zip_uses_validated_directory(tmp_path: Path) -> None:
    archive = _build_zip([{"filename": b"normal.txt", "data": bytes(range(256)) * 16}])
    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "ok"
    assert features["parse_error"] is None
    assert features["entry_count"] == 1
    assert features["eocd_count"] == 1
    assert features["lf_compression_method"] == 8
    assert features["cd_compression_method"] == 8
    assert features["method_mismatch"] is False
    assert features["suspicious_entry_count"] == 0
    assert features["lf_crc_valid"] is True


def test_lfh_cdh_method_mismatch_is_reported(tmp_path: Path) -> None:
    archive = _build_zip(
        [
            {
                "filename": b"payload.bin",
                "data": bytes(range(256)) * 32,
                "local_method": 0,
                "central_method": 8,
            }
        ]
    )
    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "ok"
    assert features["lf_compression_method"] == 0
    assert features["cd_compression_method"] == 8
    assert features["method_mismatch"] is True
    assert features["suspicious_entry_count"] == 1


def test_data_descriptor_uses_central_directory_sizes(tmp_path: Path) -> None:
    archive = _build_zip(
        [
            {
                "filename": b"streamed.txt",
                "data": b"streamed payload\n" * 40,
                "central_method": 0,
                "data_descriptor": True,
            }
        ]
    )
    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "ok"
    assert features["entry_count"] == 1
    assert features["any_crc_mismatch"] is False
    assert features["lf_crc_valid"] is True
    assert features["data_entropy_shannon"] > 0


def test_embedded_header_signatures_do_not_create_entries(tmp_path: Path) -> None:
    # Includes a structurally plausible empty EOCD as payload, as well as LFH/CDH
    # signatures. None belongs to the top-level central directory.
    payload = b"prefix" + LFH + b"middle" + CDH + EOCD + (b"\x00" * 18) + b"suffix"
    archive = _build_zip([{"filename": b"signatures.bin", "data": payload, "central_method": 0}])
    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "ok"
    assert features["entry_count"] == 1
    assert features["suspicious_entry_count"] == 0
    assert features["eocd_count"] == 1


def test_contiguous_archives_keep_validated_eocd_signal(tmp_path: Path) -> None:
    first = _build_zip([{"filename": b"first.txt", "data": b"first"}])
    second = _build_zip([{"filename": b"second.txt", "data": b"second"}])
    features = _extract(tmp_path, first + second)

    assert features["parse_status"] == "ok"
    assert features["entry_count"] == 1
    assert features["eocd_count"] == 2


def test_duplicate_filenames_are_analysed_as_distinct_entries(tmp_path: Path) -> None:
    archive = _build_zip(
        [
            {"filename": b"same.txt", "data": b"first" * 30, "central_method": 0},
            {"filename": b"same.txt", "data": b"second" * 30, "central_method": 8},
        ]
    )
    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "ok"
    assert features["entry_count"] == 2
    assert features["suspicious_entry_count"] == 0
    assert features["suspicious_entry_ratio"] == 0.0


def test_truncated_archive_is_explicitly_malformed(tmp_path: Path) -> None:
    archive = _build_zip([{"filename": b"a.txt", "data": b"content"}])
    features = _extract(tmp_path, archive[:-5])

    assert features["parse_status"] == "malformed"
    assert features["parse_error"]
    assert features["lf_crc_valid"] is False


def test_configurable_input_limit_fails_explicitly(tmp_path: Path) -> None:
    archive = _build_zip([{"filename": b"a.txt", "data": b"content"}])
    features = _extract(tmp_path, archive, max_input_size_bytes=len(archive) - 1)

    assert features["parse_status"] == "too_large"
    assert "exceeds limit" in features["parse_error"]
    assert features["file_size_bytes"] == len(archive)
    assert features["lf_crc_valid"] is False


def test_zip64_sentinel_is_explicitly_unsupported(tmp_path: Path) -> None:
    zip64_eocd = struct.pack(
        "<4sHHHHIIH",
        EOCD,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    features = _extract(tmp_path, zip64_eocd)

    assert features["parse_status"] == "unsupported"
    assert features["parse_error"] == "ZIP64 archives are not supported"


def test_encrypted_entry_is_never_scored_as_clean(tmp_path: Path) -> None:
    archive = _build_zip(
        [{"filename": b"encrypted.bin", "data": b"opaque", "flags": 0x0001}]
    )

    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "unsupported"
    assert features["parse_error"] == "encrypted ZIP entries are not supported"
    assert features["is_encrypted"] is True
    assert features["lf_crc_valid"] is False


def test_unknown_compression_method_is_never_scored_as_clean(tmp_path: Path) -> None:
    archive = _build_zip(
        [
            {
                "filename": b"unknown.bin",
                "data": b"opaque",
                "local_method": 99,
                "central_method": 99,
            }
        ]
    )

    features = _extract(tmp_path, archive)

    assert features["parse_status"] == "unsupported"
    assert features["parse_error"] == "unknown ZIP compression methods are not supported"
    assert features["lf_unknown_method"] == 1
    assert features["lf_crc_valid"] is False


def test_missing_file_is_not_a_clean_parse(tmp_path: Path) -> None:
    features = extract_features(tmp_path / "missing.zip")

    assert features["parse_status"] == "not_found"
    assert features["parse_error"]
    assert features["lf_crc_valid"] is False


def test_legacy_model_keys_remain_present(tmp_path: Path) -> None:
    features = _extract(
        tmp_path,
        _build_zip([{"filename": b"normal.txt", "data": b"hello"}]),
    )
    expected_keys = {
        "lf_compression_method",
        "cd_compression_method",
        "method_mismatch",
        "data_entropy_shannon",
        "data_entropy_renyi",
        "declared_vs_entropy_flag",
        "eocd_count",
        "lf_unknown_method",
        "entry_count",
        "suspicious_entry_count",
        "suspicious_entry_ratio",
        "entropy_variance",
        "lf_crc_valid",
        "any_crc_mismatch",
        "is_encrypted",
        "file_size_bytes",
        "parse_status",
        "parse_error",
    }

    assert expected_keys <= features.keys()


@pytest.mark.parametrize("limit", [0, -1])
def test_input_limit_must_be_positive(tmp_path: Path, limit: int) -> None:
    path = tmp_path / "sample.zip"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="must be positive"):
        extract_features(path, max_input_size_bytes=limit)
