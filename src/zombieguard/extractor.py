"""Bounded, structural ZIP feature extraction for ZombieGuard.

The central directory is the archive's entry index.  Entry payloads are reached
only through validated central-directory records and their declared local-header
offsets; signature-looking bytes inside file data are never treated as headers.
"""

from __future__ import annotations

import os
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zombieguard.entropy import compute_renyi_entropy, compute_shannon_entropy

LFH_SIGNATURE = b"PK\x03\x04"
CDH_SIGNATURE = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"

METHOD_STORE = 0
METHOD_DEFLATE = 8

# PKWARE APPNOTE compression method identifiers understood as known signals.
KNOWN_METHODS = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 18, 19, 97, 98}

ENTROPY_COMPRESSED_THRESHOLD = 7.0
DEFAULT_MAX_INPUT_SIZE_BYTES = 100 * 1024 * 1024
MAX_EOCD_SEARCH_BYTES = 65_535 + 22
MAX_VALIDATED_EOCD_CHAIN = 1_024


class _MalformedArchive(ValueError):
    """The byte stream does not form a bounded, single-disk ZIP archive."""


class _UnsupportedArchive(ValueError):
    """The archive uses a ZIP structure this extractor intentionally rejects."""


@dataclass(frozen=True)
class _EndRecord:
    offset: int
    end_offset: int
    entry_count: int
    central_size: int
    central_offset: int
    archive_base: int
    central_start: int


@dataclass(frozen=True)
class _CentralEntry:
    flags: int
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    filename_bytes: bytes
    filename: str
    local_header_offset: int


@dataclass(frozen=True)
class _LocalEntry:
    flags: int
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    filename_bytes: bytes
    filename: str
    data_offset: int

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & 0x0001)


@dataclass(frozen=True)
class _Archive:
    end_record: _EndRecord
    central_entries: tuple[_CentralEntry, ...]
    local_entries: tuple[_LocalEntry, ...]


def _feature_defaults() -> dict[str, Any]:
    """Return all legacy features plus an explicit parser outcome."""
    return {
        "lf_compression_method": -1,
        "cd_compression_method": -1,
        "method_mismatch": False,
        "data_entropy_shannon": 0.0,
        "data_entropy_renyi": 0.0,
        "declared_vs_entropy_flag": False,
        "eocd_count": 0,
        "lf_unknown_method": False,
        "entry_count": 0,
        "suspicious_entry_count": 0,
        "suspicious_entry_ratio": 0.0,
        "entropy_variance": 0.0,
        # Unknown/unparsed input must not report a successful CRC validation.
        "lf_crc_valid": False,
        "any_crc_mismatch": False,
        "is_encrypted": False,
        "file_size_bytes": 0,
        "parse_status": "unparsed",
        "parse_error": None,
    }


def _error_features(
    status: str,
    message: str,
    *,
    file_size_bytes: int = 0,
) -> dict[str, Any]:
    features = _feature_defaults()
    features["file_size_bytes"] = file_size_bytes
    features["parse_status"] = status
    features["parse_error"] = message
    return features


def _decode_filename(raw: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x0800 else "cp437"
    return raw.decode(encoding, errors="replace")


def _validate_extra_fields(extra: bytes, context: str) -> None:
    """Validate TLV boundaries and reject ZIP64 metadata explicitly."""
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise _MalformedArchive(f"truncated extra-field header in {context}")
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        field_end = cursor + field_size
        if field_end > len(extra):
            raise _MalformedArchive(f"truncated extra field in {context}")
        if field_id == 0x0001:
            raise _UnsupportedArchive("ZIP64 archives are not supported")
        cursor = field_end


def _parse_end_record(data: bytes, offset: int) -> _EndRecord:
    if offset < 0 or offset + 22 > len(data):
        raise _MalformedArchive("truncated end-of-central-directory record")

    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<4sHHHHIIH", data, offset)

    if signature != EOCD_SIGNATURE:
        raise _MalformedArchive("invalid end-of-central-directory signature")

    end_offset = offset + 22 + comment_length
    if end_offset > len(data):
        raise _MalformedArchive("truncated ZIP comment")

    if (
        entries_on_disk == 0xFFFF
        or entry_count == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise _UnsupportedArchive("ZIP64 archives are not supported")

    if disk_number != 0 or central_disk != 0 or entries_on_disk != entry_count:
        raise _UnsupportedArchive("multi-disk ZIP archives are not supported")

    if central_size > offset:
        raise _MalformedArchive("central directory extends before the input")

    central_start = offset - central_size
    archive_base = central_start - central_offset
    if archive_base < 0:
        raise _MalformedArchive("invalid central-directory offset")

    return _EndRecord(
        offset=offset,
        end_offset=end_offset,
        entry_count=entry_count,
        central_size=central_size,
        central_offset=central_offset,
        archive_base=archive_base,
        central_start=central_start,
    )


def _parse_central_directory(
    data: bytes,
    end_record: _EndRecord,
) -> tuple[_CentralEntry, ...]:
    entries: list[_CentralEntry] = []
    cursor = end_record.central_start
    central_end = end_record.offset

    for entry_index in range(end_record.entry_count):
        if cursor + 46 > central_end:
            raise _MalformedArchive(f"truncated central-directory header for entry {entry_index}")

        fields = struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, cursor)
        if fields[0] != CDH_SIGNATURE:
            raise _MalformedArchive(f"invalid central-directory signature for entry {entry_index}")

        flags = fields[3]
        method = fields[4]
        crc32 = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        disk_start = fields[13]
        local_header_offset = fields[16]

        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
            or disk_start == 0xFFFF
        ):
            raise _UnsupportedArchive("ZIP64 archives are not supported")
        if disk_start != 0:
            raise _UnsupportedArchive("multi-disk ZIP archives are not supported")

        record_end = cursor + 46 + filename_length + extra_length + comment_length
        if record_end > central_end:
            raise _MalformedArchive(
                f"central-directory entry {entry_index} exceeds its declared bounds"
            )

        filename_start = cursor + 46
        filename_end = filename_start + filename_length
        extra_end = filename_end + extra_length
        filename_bytes = data[filename_start:filename_end]
        _validate_extra_fields(
            data[filename_end:extra_end],
            f"central-directory entry {entry_index}",
        )

        entries.append(
            _CentralEntry(
                flags=flags,
                compression_method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                filename_bytes=filename_bytes,
                filename=_decode_filename(filename_bytes, flags),
                local_header_offset=local_header_offset,
            )
        )
        cursor = record_end

    if cursor != central_end:
        raise _UnsupportedArchive("unsupported records are present inside the central directory")

    return tuple(entries)


def _parse_local_entry(
    data: bytes,
    end_record: _EndRecord,
    central: _CentralEntry,
    entry_index: int,
) -> _LocalEntry:
    offset = end_record.archive_base + central.local_header_offset
    if offset < end_record.archive_base or offset + 30 > end_record.central_start:
        raise _MalformedArchive(f"invalid local-header offset for entry {entry_index}")

    fields = struct.unpack_from("<4sHHHHHIIIHH", data, offset)
    if fields[0] != LFH_SIGNATURE:
        raise _MalformedArchive(f"missing local-file header for entry {entry_index}")

    flags = fields[2]
    method = fields[3]
    crc32 = fields[6]
    compressed_size = fields[7]
    uncompressed_size = fields[8]
    filename_length = fields[9]
    extra_length = fields[10]
    filename_start = offset + 30
    filename_end = filename_start + filename_length
    data_offset = filename_end + extra_length

    if data_offset > end_record.central_start:
        raise _MalformedArchive(f"local-file header {entry_index} exceeds archive bounds")

    filename_bytes = data[filename_start:filename_end]
    _validate_extra_fields(
        data[filename_end:data_offset],
        f"local-file header {entry_index}",
    )

    # Compressed size is deliberately not taken from this header.  With general
    # purpose bit 3 it is normally zero and the data descriptor follows payload.
    return _LocalEntry(
        flags=flags,
        compression_method=method,
        crc32=crc32,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        filename_bytes=filename_bytes,
        filename=_decode_filename(filename_bytes, flags),
        data_offset=data_offset,
    )


def _parse_archive(data: bytes, eocd_offset: int) -> _Archive:
    end_record = _parse_end_record(data, eocd_offset)
    central_entries = _parse_central_directory(data, end_record)

    local_offsets = [
        end_record.archive_base + entry.local_header_offset for entry in central_entries
    ]
    if len(local_offsets) != len(set(local_offsets)):
        raise _MalformedArchive("multiple central-directory entries reference one local header")

    local_entries = tuple(
        _parse_local_entry(data, end_record, central, index)
        for index, central in enumerate(central_entries)
    )

    # Entry data may be followed by a data descriptor, so only require each
    # payload to finish before the next declared header (or central directory).
    sorted_offsets = sorted(local_offsets)
    next_boundary = {
        offset: (
            sorted_offsets[index + 1]
            if index + 1 < len(sorted_offsets)
            else end_record.central_start
        )
        for index, offset in enumerate(sorted_offsets)
    }
    for index, (central, local) in enumerate(
        zip(central_entries, local_entries, strict=True)
    ):
        physical_offset = end_record.archive_base + central.local_header_offset
        payload_end = local.data_offset + central.compressed_size
        if payload_end > next_boundary[physical_offset]:
            raise _MalformedArchive(f"compressed payload {index} exceeds archive bounds")

    return _Archive(end_record, central_entries, local_entries)


def _signature_positions(
    data: bytes,
    signature: bytes,
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    """Locate EOCD candidates only; entry headers are never signature-scanned."""
    positions: list[int] = []
    cursor = max(start, 0)
    stop = len(data) if end is None else min(end, len(data))
    while True:
        position = data.find(signature, cursor, stop)
        if position < 0:
            return positions
        positions.append(position)
        cursor = position + 1


def _find_terminal_archive(data: bytes) -> _Archive:
    search_start = max(0, len(data) - MAX_EOCD_SEARCH_BYTES)
    positions = _signature_positions(data, EOCD_SIGNATURE, search_start)
    terminal_positions: list[int] = []

    for position in positions:
        if position + 22 > len(data):
            continue
        comment_length = struct.unpack_from("<H", data, position + 20)[0]
        if position + 22 + comment_length == len(data):
            terminal_positions.append(position)

    if not terminal_positions:
        if (
            ZIP64_EOCD_SIGNATURE in data[search_start:]
            or ZIP64_LOCATOR_SIGNATURE in data[search_start:]
        ):
            raise _UnsupportedArchive("ZIP64 archives are not supported")
        raise _MalformedArchive("a terminal end-of-central-directory record was not found")

    first_error: ValueError | None = None
    # The actual EOCD precedes its comment, so prefer the earliest structurally
    # valid candidate when signature bytes also occur inside that comment.
    for position in terminal_positions:
        try:
            return _parse_archive(data, position)
        except (_MalformedArchive, _UnsupportedArchive) as exc:
            if first_error is None:
                first_error = exc

    if first_error is None:  # Defensive: terminal_positions cannot make this reachable.
        raise _MalformedArchive("no structurally valid terminal ZIP record was found")
    raise first_error


def _validated_eocd_count(data: bytes, primary: _Archive) -> int:
    """Count a contiguous chain of structurally valid top-level ZIP records.

    Nested archives and EOCD-looking payload bytes are excluded because their
    records do not end exactly where the next top-level archive begins.
    """
    count = 1
    chain_start = primary.end_record.archive_base
    seen_offsets = {primary.end_record.offset}
    while chain_start > 0 and count < MAX_VALIDATED_EOCD_CHAIN:
        search_start = max(0, chain_start - MAX_EOCD_SEARCH_BYTES)
        candidates: list[_Archive] = []
        for position in _signature_positions(data, EOCD_SIGNATURE, search_start, chain_start):
            if position >= chain_start or position + 22 > chain_start:
                continue
            comment_length = struct.unpack_from("<H", data, position + 20)[0]
            if position + 22 + comment_length != chain_start:
                continue
            try:
                archive = _parse_archive(data, position)
            except (_MalformedArchive, _UnsupportedArchive):
                continue
            if (
                archive.end_record.offset not in seen_offsets
                and archive.end_record.archive_base < chain_start
            ):
                candidates.append(archive)
        if not candidates:
            break
        predecessor = max(candidates, key=lambda item: item.end_record.archive_base)
        seen_offsets.add(predecessor.end_record.offset)
        chain_start = predecessor.end_record.archive_base
        count += 1

    return count


def _analyse_archive(data: bytes, archive: _Archive) -> dict[str, Any]:
    entry_entropies: list[float] = []
    suspicious_count = 0
    any_encrypted = False
    any_unknown_method = False
    any_crc_mismatch = False
    first_crc_valid = True

    for central, local in zip(
        archive.central_entries, archive.local_entries, strict=True
    ):
        entry_suspicious = False
        uses_data_descriptor = bool((central.flags | local.flags) & 0x0008)
        entry_encrypted = bool((central.flags | local.flags) & 0x0001)
        any_encrypted = any_encrypted or entry_encrypted

        if (
            local.compression_method not in KNOWN_METHODS
            or central.compression_method not in KNOWN_METHODS
        ):
            any_unknown_method = True

        if local.compression_method != central.compression_method:
            entry_suspicious = True
        if local.filename_bytes != central.filename_bytes:
            entry_suspicious = True
        if local.flags != central.flags:
            entry_suspicious = True

        if not uses_data_descriptor:
            if local.crc32 != central.crc32:
                any_crc_mismatch = True
                entry_suspicious = True
            if (
                local.compressed_size != central.compressed_size
                or local.uncompressed_size != central.uncompressed_size
            ):
                entry_suspicious = True

        payload = data[local.data_offset : local.data_offset + central.compressed_size]
        if payload:
            entry_entropies.append(compute_shannon_entropy(payload))

        # CRC can be checked without decompression only when both headers declare
        # STORE.  The central-directory CRC is authoritative with bit 3 set.
        if (
            not entry_encrypted
            and local.compression_method == METHOD_STORE
            and central.compression_method == METHOD_STORE
        ):
            actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if actual_crc != central.crc32:
                any_crc_mismatch = True
                entry_suspicious = True

        if entry_suspicious:
            suspicious_count += 1

    entry_count = len(archive.central_entries)
    if entry_count and any_crc_mismatch:
        first_central = archive.central_entries[0]
        first_local = archive.local_entries[0]
        first_uses_descriptor = bool((first_central.flags | first_local.flags) & 0x0008)
        if not first_uses_descriptor and first_local.crc32 != first_central.crc32:
            first_crc_valid = False
        elif (
            first_local.compression_method == METHOD_STORE
            and first_central.compression_method == METHOD_STORE
            and not bool((first_central.flags | first_local.flags) & 0x0001)
        ):
            payload = data[
                first_local.data_offset : first_local.data_offset + first_central.compressed_size
            ]
            first_crc_valid = (zlib.crc32(payload) & 0xFFFFFFFF) == first_central.crc32

    return {
        "entry_count": entry_count,
        "suspicious_entry_count": suspicious_count,
        "suspicious_entry_ratio": (
            round(suspicious_count / entry_count, 4) if entry_count else 0.0
        ),
        "max_entropy_shannon": (round(max(entry_entropies), 4) if entry_entropies else 0.0),
        "entropy_variance": (
            round(statistics.pvariance(entry_entropies), 4) if len(entry_entropies) > 1 else 0.0
        ),
        "any_encrypted": any_encrypted,
        "any_unknown_method": any_unknown_method,
        "any_crc_mismatch": any_crc_mismatch,
        "lf_crc_valid": first_crc_valid,
    }


def extract_features(
    zip_filepath: str | os.PathLike[str],
    max_input_size_bytes: int = DEFAULT_MAX_INPUT_SIZE_BYTES,
) -> dict[str, Any]:
    """Extract legacy model features and a fail-explicit parser outcome.

    ``max_input_size_bytes`` bounds both allocation and analysis work.  Parsing
    failures return ``parse_status``/``parse_error`` and never claim a valid CRC.
    Supported statuses are ``ok``, ``not_found``, ``io_error``, ``too_large``,
    ``malformed``, and ``unsupported``.
    """
    if isinstance(max_input_size_bytes, bool) or not isinstance(max_input_size_bytes, int):
        raise TypeError("max_input_size_bytes must be an integer")
    if max_input_size_bytes <= 0:
        raise ValueError("max_input_size_bytes must be positive")

    path = Path(zip_filepath)
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        return _error_features("not_found", f"input file does not exist: {path}")
    except OSError as exc:
        return _error_features("io_error", f"unable to stat input: {exc}")

    if not path.is_file():
        return _error_features(
            "io_error",
            f"input is not a regular file: {path}",
            file_size_bytes=file_size,
        )

    if file_size > max_input_size_bytes:
        return _error_features(
            "too_large",
            f"input size {file_size} exceeds limit {max_input_size_bytes}",
            file_size_bytes=file_size,
        )

    try:
        with path.open("rb") as archive_file:
            data = archive_file.read(max_input_size_bytes + 1)
    except OSError as exc:
        return _error_features(
            "io_error",
            f"unable to read input: {exc}",
            file_size_bytes=file_size,
        )

    # Re-check the bound after reading to cover a file growing after stat().
    if len(data) > max_input_size_bytes:
        return _error_features(
            "too_large",
            f"input exceeds limit {max_input_size_bytes} while being read",
            file_size_bytes=max(file_size, len(data)),
        )

    try:
        archive = _find_terminal_archive(data)
    except _UnsupportedArchive as exc:
        return _error_features("unsupported", str(exc), file_size_bytes=len(data))
    except _MalformedArchive as exc:
        return _error_features("malformed", str(exc), file_size_bytes=len(data))

    analysis = _analyse_archive(data, archive)
    features = _feature_defaults()
    features.update(
        {
            "parse_status": "ok",
            "parse_error": None,
            "file_size_bytes": len(data),
            "eocd_count": _validated_eocd_count(data, archive),
            "entry_count": analysis["entry_count"],
            "suspicious_entry_count": analysis["suspicious_entry_count"],
            "suspicious_entry_ratio": analysis["suspicious_entry_ratio"],
            "entropy_variance": analysis["entropy_variance"],
            "any_crc_mismatch": analysis["any_crc_mismatch"],
            "lf_crc_valid": analysis["lf_crc_valid"],
            "is_encrypted": analysis["any_encrypted"],
            "lf_unknown_method": int(analysis["any_unknown_method"]),
        }
    )

    if not archive.central_entries:
        return features

    first_central = archive.central_entries[0]
    first_local = archive.local_entries[0]
    features["lf_compression_method"] = first_local.compression_method
    features["cd_compression_method"] = first_central.compression_method
    features["method_mismatch"] = first_local.compression_method != first_central.compression_method
    features["data_entropy_shannon"] = analysis["max_entropy_shannon"]

    first_payload = data[
        first_local.data_offset : first_local.data_offset + first_central.compressed_size
    ]
    if first_payload:
        features["data_entropy_renyi"] = round(compute_renyi_entropy(first_payload, alpha=2.0), 4)

    features["declared_vs_entropy_flag"] = bool(
        first_local.compression_method == METHOD_STORE
        and features["data_entropy_shannon"] > ENTROPY_COMPRESSED_THRESHOLD
    )

    # Structural metadata can still be inspected for these inputs, but their
    # payload semantics are outside the supported scanner contract. Never pass
    # them to the model or allow them to become a clean verdict.
    if features["is_encrypted"]:
        features["parse_status"] = "unsupported"
        features["parse_error"] = "encrypted ZIP entries are not supported"
        features["lf_crc_valid"] = False
    elif features["lf_unknown_method"]:
        features["parse_status"] = "unsupported"
        features["parse_error"] = "unknown ZIP compression methods are not supported"
        features["lf_crc_valid"] = False
    return features
