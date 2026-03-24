"""
generate_zombie_samples.py
Generates diverse synthetic Zombie ZIP samples across 8 structural variant types.
Designed to mirror real-world attacker behavior documented in:
- CVE-2026-0866 (Zombie ZIP)
- Gootloader campaign analysis
- BadPack APK research (Palo Alto Unit42)
"""

import os
import struct
import zlib
import bz2
import random
import string
import hashlib
from pathlib import Path

OUTPUT_DIR = "data/raw/malicious"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- ZIP constants ------------------------------------------------------------
SIG_LFH = b"PK\x03\x04"
SIG_CDH = b"PK\x01\x02"
SIG_EOCD = b"PK\x05\x06"

METHOD_STORE = 0
METHOD_DEFLATE = 8
METHOD_BZIP2 = 12
METHOD_LZMA = 14


# -- Low-level ZIP block builders --------------------------------------------

def _lfh(
    method: int,
    crc: int,
    comp_size: int,
    uncomp_size: int,
    filename: bytes,
    extra: bytes = b"",
) -> bytes:
    return struct.pack(
        "<4sHHHHHIIIHH",
        SIG_LFH,
        20,
        0,
        method,
        0,
        0,
        crc,
        comp_size,
        uncomp_size,
        len(filename),
        len(extra),
    ) + filename + extra


def _cdh(
    method: int,
    crc: int,
    comp_size: int,
    uncomp_size: int,
    filename: bytes,
    lfh_offset: int,
    extra: bytes = b"",
) -> bytes:
    return struct.pack(
        "<4sHHHHHHIIIHHHHHII",
        SIG_CDH,
        20,
        20,
        0,
        method,
        0,
        0,
        crc,
        comp_size,
        uncomp_size,
        len(filename),
        len(extra),
        0,
        0,
        0,
        0,
        lfh_offset,
    ) + filename + extra


def _eocd(num_entries: int, cd_size: int, cd_offset: int) -> bytes:
    return struct.pack(
        "<4sHHHHIIH",
        SIG_EOCD,
        0,
        0,
        num_entries,
        num_entries,
        cd_size,
        cd_offset,
        0,
    )


def _compress_deflate(data: bytes, level: int = 6) -> bytes:
    """Strip zlib header/trailer to get raw DEFLATE bytes."""
    return zlib.compress(data, level)[2:-4]


def _compress_bzip2(data: bytes) -> bytes:
    return bz2.compress(data)


# -- Payload generators -------------------------------------------------------

def payload_eicar(variant: int = 0) -> bytes:
    """EICAR-style test string with variant suffix."""
    base = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$ZOMBIE-TEST-FILE!$H+H*"
    return base + f"_v{variant}".encode()


def payload_random_binary(size: int) -> bytes:
    """High-entropy random bytes - simulates encrypted/packed malware."""
    return os.urandom(size)


def payload_repetitive(size: int) -> bytes:
    """Low-entropy repetitive bytes - compresses very well, high mismatch signal."""
    unit = random.choice([b"AAAA", b"\x00\x01\x02\x03", b"MZ\x90\x00"])
    return (unit * (size // len(unit) + 1))[:size]


def payload_pe_header_lure() -> bytes:
    """Starts with MZ header (PE executable signature) followed by random bytes."""
    mz = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xFF\xFF\x00\x00"
    return mz + os.urandom(random.randint(512, 4096))


def payload_script_lure() -> bytes:
    """Looks like obfuscated script content."""
    chars = string.ascii_letters + string.digits + "{}[]<>()=+-*/&|^%$#@!"
    content = "".join(random.choices(chars, k=random.randint(800, 3000)))
    return content.encode()


def payload_document_lure() -> bytes:
    """Simulates document content - realistic for phishing ZIPs."""
    words = [
        "invoice",
        "payment",
        "confidential",
        "salary",
        "report",
        "meeting",
        "contract",
        "updated",
        "urgent",
        "review",
    ]
    content = " ".join(random.choices(words, k=random.randint(100, 500)))
    return content.encode() * random.randint(3, 15)


def random_filename(ext: str = ".bin") -> bytes:
    """Generate realistic-looking filenames."""
    prefixes = [
        "invoice_2026",
        "salary_sheet",
        "meeting_notes",
        "contract_draft",
        "payload",
        "update",
        "patch",
        "installer",
        "readme",
        "document",
        "report_Q1",
        "data_export",
        "backup",
        "config",
        "setup",
    ]
    name = random.choice(prefixes) + f"_{random.randint(100,9999)}" + ext
    return name.encode()


def add_decoy_entries_raw(zombie_zip_bytes: bytes, count: int = None) -> bytes:
    """
    Append benign decoy entries at the binary level without rewriting originals.

    This preserves malicious LFH/CDH mismatches by avoiding zipfile-based copying.
    """
    if count is None:
        count = random.randint(5, 50)

    eocd_pos = zombie_zip_bytes.rfind(SIG_EOCD)
    if eocd_pos == -1:
        return zombie_zip_bytes

    try:
        eocd_fields = struct.unpack_from("<4sHHHHIIH", zombie_zip_bytes, eocd_pos)
    except struct.error:
        return zombie_zip_bytes

    _, _, _, _, entries_total, _, cd_offset, _ = eocd_fields

    if cd_offset > len(zombie_zip_bytes) or cd_offset > eocd_pos:
        return zombie_zip_bytes

    local_section = zombie_zip_bytes[:cd_offset]
    cd_section = zombie_zip_bytes[cd_offset:eocd_pos]

    decoy_filenames = [
        "readme.txt",
        "license.txt",
        "manifest.json",
        "config.ini",
        "version.txt",
        "changelog.md",
        "docs/index.html",
        "assets/style.css",
        "src/main.py",
        "src/utils.py",
        "src/config.py",
        "tests/test_main.py",
        "data/sample.csv",
        "build.gradle",
        "pom.xml",
        ".gitignore",
        "Makefile",
        "setup.py",
        "pyproject.toml",
    ]
    random.shuffle(decoy_filenames)
    selected = decoy_filenames[:min(count, len(decoy_filenames))]

    new_local_entries = b""
    new_cd_entries = b""

    for fname_str in selected:
        fname = fname_str.encode()
        content = (
            " ".join(
                random.choices(
                    [
                        "config",
                        "version",
                        "path",
                        "name",
                        "value",
                        "type",
                        "source",
                        "target",
                        "module",
                        "class",
                    ],
                    k=random.randint(10, 80),
                )
            )
        ).encode()

        compressed = _compress_deflate(content)
        crc = zlib.crc32(content) & 0xFFFFFFFF
        offset = len(local_section) + len(new_local_entries)

        lfh = struct.pack(
            "<4sHHHHHIIIHH",
            SIG_LFH,
            20,
            0,
            METHOD_DEFLATE,
            0,
            0,
            crc,
            len(compressed),
            len(content),
            len(fname),
            0,
        ) + fname + compressed

        cdh = struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            SIG_CDH,
            20,
            20,
            0,
            METHOD_DEFLATE,
            0,
            0,
            crc,
            len(compressed),
            len(content),
            len(fname),
            0,
            0,
            0,
            0,
            0,
            offset,
        ) + fname

        new_local_entries += lfh
        new_cd_entries += cdh

    new_local_section = local_section + new_local_entries
    new_cd_section = cd_section + new_cd_entries
    new_cd_offset = len(new_local_section)
    new_total_entries = entries_total + len(selected)

    new_eocd = struct.pack(
        "<4sHHHHIIH",
        SIG_EOCD,
        0,
        0,
        new_total_entries,
        new_total_entries,
        len(new_cd_section),
        new_cd_offset,
        0,
    )

    return new_local_section + new_cd_section + new_eocd


# -- Variant builders ---------------------------------------------------------

def variant_classic_zombie(filepath: str):
    """
    VARIANT A - Classic Zombie ZIP (CVE-2026-0866 original)
    LFH declares STORE (0), CDH declares DEFLATE (8).
    Actual data is DEFLATE compressed.
    This is the original attack described in the CVE.
    """
    payload = random.choice(
        [
            payload_eicar(random.randint(0, 99)),
            payload_repetitive(random.randint(512, 8192)),
            payload_document_lure(),
            payload_script_lure(),
        ]
    )
    compressed = _compress_deflate(payload, level=random.randint(1, 9))
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    fname = random_filename(random.choice([".bin", ".exe", ".dll", ".js"]))

    local = _lfh(METHOD_STORE, crc, len(compressed), len(payload), fname)
    local += compressed
    cd_offset = len(local)
    cd = _cdh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


def variant_method_only_mismatch(filepath: str):
    """
    VARIANT B - Method mismatch only, data actually stored
    LFH declares DEFLATE (8), CDH declares STORE (0).
    Actual data is NOT compressed - tests method_mismatch feature alone.
    Entropy will be low but method fields disagree.
    """
    payload = payload_document_lure()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    fname = random_filename(".txt")

    local = _lfh(METHOD_DEFLATE, crc, len(payload), len(payload), fname)
    local += payload
    cd_offset = len(local)
    cd = _cdh(METHOD_STORE, crc, len(payload), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


def variant_gootloader_concat(filepath: str, chain_len: int = None):
    """
    VARIANT C - Gootloader-style concatenated ZIPs
    Multiple valid ZIPs concatenated - EOCD count > 1.
    Security tools only parse the first ZIP and miss payloads in later ones.
    Based on: in-the-wild Gootloader campaign analysis (March 2026).
    """
    if chain_len is None:
        chain_len = random.randint(2, 8)

    blob = b""
    for i in range(chain_len):
        if i == chain_len - 1:
            # Last ZIP contains the actual malicious payload
            payload = payload_pe_header_lure()
            compressed = _compress_deflate(payload)
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            fname = random_filename(".exe")
            local = _lfh(METHOD_STORE, crc, len(compressed), len(payload), fname)
            local += compressed
        else:
            # Decoy ZIPs contain benign-looking content
            payload = payload_document_lure()
            compressed = _compress_deflate(payload)
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            fname = random_filename(".txt")
            local = _lfh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname)
            local += compressed

        cd_offset = len(local)
        cd = _cdh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname, 0)
        eocd_block = _eocd(1, len(cd), cd_offset)
        blob += local + cd + eocd_block

    with open(filepath, "wb") as file:
        file.write(blob)


def variant_multi_file_zip(filepath: str):
    """
    VARIANT D - Multi-file ZIP with one malicious entry hidden among benign files
    Realistic delivery: attacker includes decoy files to look legitimate.
    Only the hidden payload file has the header mismatch.
    """
    entries_local = b""
    entries_cd = b""
    offsets = []
    num_decoys = random.randint(2, 5)

    # Decoy files - normal, benign-looking
    for i in range(num_decoys):
        payload = payload_document_lure()
        compressed = _compress_deflate(payload)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        fname = random_filename(random.choice([".txt", ".pdf", ".docx"]))
        offsets.append(len(entries_local))
        local_block = _lfh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname)
        entries_local += local_block + compressed
        entries_cd += _cdh(
            METHOD_DEFLATE,
            crc,
            len(compressed),
            len(payload),
            fname,
            offsets[-1],
        )

    # Hidden malicious file - Zombie ZIP pattern
    mal_payload = payload_pe_header_lure()
    mal_compressed = _compress_deflate(mal_payload)
    mal_crc = zlib.crc32(mal_payload) & 0xFFFFFFFF
    mal_fname = random_filename(".bin")
    offsets.append(len(entries_local))
    mal_local = _lfh(METHOD_STORE, mal_crc, len(mal_compressed), len(mal_payload), mal_fname)
    entries_local += mal_local + mal_compressed
    entries_cd += _cdh(
        METHOD_DEFLATE,
        mal_crc,
        len(mal_compressed),
        len(mal_payload),
        mal_fname,
        offsets[-1],
    )

    cd_offset = len(entries_local)
    total_entries = num_decoys + 1
    eocd_block = _eocd(total_entries, len(entries_cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(entries_local + entries_cd + eocd_block)


def variant_crc_mismatch(filepath: str):
    """
    VARIANT E - CRC32 mismatch between LFH and CDH
    LFH CRC is deliberately wrong. CDH has correct CRC.
    Some parsers trust LFH CRC for validation - this bypasses them.
    """
    payload = payload_random_binary(random.randint(256, 2048))
    real_crc = zlib.crc32(payload) & 0xFFFFFFFF
    fake_crc = (real_crc ^ 0xDEADBEEF) & 0xFFFFFFFF  # deliberately corrupted
    compressed = _compress_deflate(payload)
    fname = random_filename(".bin")

    local = _lfh(METHOD_STORE, fake_crc, len(compressed), len(payload), fname)
    local += compressed
    cd_offset = len(local)
    cd = _cdh(METHOD_DEFLATE, real_crc, len(compressed), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


def variant_extra_field_noise(filepath: str):
    """
    VARIANT F - Junk bytes in extra field to confuse parser offsets
    Extra field in LFH is padded with random bytes.
    Some parsers miscalculate data offset and scan wrong bytes.
    Zombie ZIP pattern preserved in method fields.
    """
    payload = payload_repetitive(random.randint(1024, 4096))
    compressed = _compress_deflate(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    fname = random_filename(".dat")
    extra = os.urandom(random.randint(8, 64))  # junk extra field

    local = _lfh(METHOD_STORE, crc, len(compressed), len(payload), fname, extra)
    local += compressed
    cd_offset = len(local)
    cd = _cdh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


def variant_high_compression_mismatch(filepath: str):
    """
    VARIANT G - Highly compressible payload with maximum compression
    Declares STORE but uses zlib level=9.
    Creates maximum entropy gap between declared and actual.
    """
    # Very compressible data - long repeating patterns
    unit = random.choice([b"\x00" * 16, b"AAAAAAAAAAAAAAAA", b"\xFF\x00" * 8, b"MZ" * 8])
    payload = unit * random.randint(200, 1000)
    compressed = _compress_deflate(payload, level=9)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    fname = random_filename(".bin")

    local = _lfh(METHOD_STORE, crc, len(compressed), len(payload), fname)
    local += compressed
    cd_offset = len(local)
    cd = _cdh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


def variant_size_field_mismatch(filepath: str):
    """
    VARIANT H - Compressed size field mismatch between LFH and CDH
    LFH declares wrong compressed size - CDH has correct size.
    Tools that rely on LFH size to determine scan boundary miss the payload.
    """
    payload = payload_script_lure()
    compressed = _compress_deflate(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    fname = random_filename(".js")

    fake_size = len(compressed) + random.randint(10, 100)  # wrong in LFH

    local = _lfh(METHOD_STORE, crc, fake_size, len(payload), fname)
    local += compressed
    cd_offset = len(local)
    # CDH has correct size
    cd = _cdh(METHOD_DEFLATE, crc, len(compressed), len(payload), fname, 0)
    eocd = _eocd(1, len(cd), cd_offset)

    with open(filepath, "wb") as file:
        file.write(local + cd + eocd)


# -- Master generator ---------------------------------------------------------

VARIANTS = [
    ("A_classic", variant_classic_zombie, 350),
    ("B_method_only", variant_method_only_mismatch, 100),
    ("C_gootloader", variant_gootloader_concat, 150),
    ("D_multifile", variant_multi_file_zip, 150),
    ("E_crc_mismatch", variant_crc_mismatch, 100),
    ("F_extra_noise", variant_extra_field_noise, 100),
    ("G_high_compression", variant_high_compression_mismatch, 100),
    ("H_size_mismatch", variant_size_field_mismatch, 100),
]
# Total: 1150 samples across 8 distinct structural variants


def generate_all():
    # Clear existing synthetic malicious samples
    existing = list(Path(OUTPUT_DIR).glob("zombie_*.zip"))
    for file in existing:
        file.unlink()
    print(f"Cleared {len(existing)} old samples")

    total = 0
    for variant_name, func, count in VARIANTS:
        print(f"Generating variant {variant_name}: {count} samples...")
        for i in range(count):
            filepath = os.path.join(OUTPUT_DIR, f"zombie_{variant_name}_{i:04d}.zip")
            try:
                func(filepath)

                # Add decoy entries to normalize entry-count distribution.
                with open(filepath, "rb") as file:
                    raw = file.read()
                normalized = add_decoy_entries_raw(raw)
                with open(filepath, "wb") as file:
                    file.write(normalized)

                total += 1
            except Exception as exc:
                print(f"  ERROR at {filepath}: {exc}")

    print(f"\nTotal generated: {total} samples across {len(VARIANTS)} variants")
    print(f"Saved to: {OUTPUT_DIR}")

    # Print variant breakdown
    print("\nVariant breakdown:")
    for name, _, count in VARIANTS:
        print(f"  {name:<30} {count} samples")


if __name__ == "__main__":
    generate_all()
