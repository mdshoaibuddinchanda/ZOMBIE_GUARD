"""
build_hard_testset.py
Builds a harder evaluation test set where eocd_count alone
cannot separate classes. Forces models to use entropy,
method mismatch, CRC, and structural features.

Test set composition:
  Positive (evasion):
    14 real evasion samples (from real_splits/test/evasion)
    14 synthetic - variants B, E, F, H only (no EOCD signal)
    Total: 28 evasion samples

  Negative (non-evasion):
    189 real non-evasion malicious ZIPs
    50 hard negative benign ZIPs (structural quirks)
    Total: 239 non-evasion samples

Grand total: 267 samples
"""

import os
import sys
import random
import shutil
import struct
import zlib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
from src.extractor import extract_features

HARD_TEST_DIR = "data/hard_test"
HARD_EV_DIR = os.path.join(HARD_TEST_DIR, "evasion")
HARD_NEV_DIR = os.path.join(HARD_TEST_DIR, "non_evasion")

REAL_TEST_EV = "data/real_splits/test/evasion"
REAL_TEST_NEV = "data/real_splits/test/non_evasion"
SYNTH_MAL_DIR = "data/raw/malicious"
BENIGN_DIR = "data/raw/benign"

for d in [HARD_EV_DIR, HARD_NEV_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(42)

SIG_LFH = b'PK\x03\x04'
SIG_CDH = b'PK\x01\x02'
SIG_EOCD = b'PK\x05\x06'

# Files to exclude from hard-test real evasion pool (trivial high-EOCD cases).
GOOTLOADER_EXCLUDE = {
    "0f66bcdf70104d59e87e.zip",  # eocd=432
    "36b76204518245f15bed.zip",  # eocd=618
    "5d6433cc03d11daac1bb.zip",  # eocd=365
    "9a5409c52b1a4e878ca2.zip",  # eocd=990
}


def compress_deflate(data: bytes) -> bytes:
    return zlib.compress(data)[2:-4]


# -- Step 1: Copy real evasion test samples --
def copy_real_evasion():
    """
    Copy only non-Gootloader real evasion samples.
    Excludes high-EOCD concatenated archives that are trivially detectable.
    """
    files = [f for f in os.listdir(REAL_TEST_EV)
             if f.endswith('.zip')
             and f not in GOOTLOADER_EXCLUDE]
    copied = 0
    for fname in files:
        src = os.path.join(REAL_TEST_EV, fname)
        dst = os.path.join(HARD_EV_DIR, f"real_{fname}")
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} non-Gootloader real evasion samples")
    print(f"  (excluded {len(GOOTLOADER_EXCLUDE)} high-EOCD Gootloader samples)")
    return copied


# -- Step 2: Select synthetic variants B, E, F, H --
def copy_synthetic_hard_variants(count: int = 14):
    """
    Select synthetic samples from variants B, E, F, H only.
    These have EOCD=1 - no concatenation signal.
    Forces model to use method, CRC, entropy features.
    """
    target_prefixes = [
        "zombie_B_method_only_",
        "zombie_E_crc_mismatch_",
        "zombie_F_extra_noise_",
        "zombie_H_size_mismatch_",
    ]

    candidates = []
    for fname in os.listdir(SYNTH_MAL_DIR):
        if not fname.endswith('.zip'):
            continue
        for prefix in target_prefixes:
            if fname.startswith(prefix):
                candidates.append(fname)
                break

    random.shuffle(candidates)
    selected = candidates[:count]

    copied = 0
    for fname in selected:
        src = os.path.join(SYNTH_MAL_DIR, fname)
        dst = os.path.join(HARD_EV_DIR, f"synth_{fname}")
        shutil.copy2(src, dst)
        copied += 1

    print(f"Copied {copied} synthetic hard-variant samples")
    print(f"  (from {len(candidates)} available B/E/F/H variants)")
    return copied


# -- Step 3: Copy real non-evasion samples --
def copy_real_non_evasion():
    files = [f for f in os.listdir(REAL_TEST_NEV)
             if f.endswith('.zip')]
    copied = 0
    for fname in files:
        src = os.path.join(REAL_TEST_NEV, fname)
        dst = os.path.join(HARD_NEV_DIR, f"real_{fname}")
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} real non-evasion samples")
    return copied


# -- Step 4: Generate hard negative benign ZIPs --
def generate_hard_negatives(count: int = 50):
    """
    Benign ZIPs with structural quirks that look
    superficially suspicious but are not malicious.
    Tests false positive rate on edge cases.
    """
    generated = 0

    quirk_types = [
        'high_entropy_stored',
        'multi_entry_varied',
        'old_compression_method',
        'zero_crc',
        'large_extra_field',
    ]

    for i in range(count):
        quirk = quirk_types[i % len(quirk_types)]
        fname = f"hard_neg_{quirk}_{i:04d}.zip"
        fpath = os.path.join(HARD_NEV_DIR, fname)

        content = os.urandom(
            random.randint(512, 4096)
        ) if quirk == 'high_entropy_stored' else (
            ' '.join(['word'] * random.randint(100, 500))
        ).encode()

        if quirk == 'high_entropy_stored':
            # High entropy but STORED - not compressed.
            # Tests: high entropy alone should NOT trigger.
            crc = zlib.crc32(content) & 0xFFFFFFFF
            method = 0  # STORE
            data = content  # actually stored

        elif quirk == 'multi_entry_varied':
            # Multiple entries - tests entry_count signal.
            buf = _build_multi_entry_zip(
                random.randint(5, 20)
            )
            with open(fpath, 'wb') as f:
                f.write(buf)
            generated += 1
            continue

        elif quirk == 'old_compression_method':
            # Method 6 = implode (old but valid)
            crc = zlib.crc32(content) & 0xFFFFFFFF
            method = 6
            data = content

        elif quirk == 'zero_crc':
            # CRC=0 in both LFH and CDH.
            # Some tools legitimately write this.
            crc = 0
            method = 8
            data = compress_deflate(content)

        else:  # large_extra_field
            crc = zlib.crc32(content) & 0xFFFFFFFF
            method = 8
            data = compress_deflate(content)

        _write_single_entry_zip(
            fpath, method, crc, data, content, b"file.bin"
        )
        generated += 1

    print(f"Generated {generated} hard negative benign ZIPs")
    return generated


def _write_single_entry_zip(fpath, method, crc,
                            compressed, original,
                            filename):
    """Write a minimal valid ZIP with one entry."""
    fname = filename

    extra = b""
    if method == 6:
        extra = b"\x00" * 4

    lfh = struct.pack('<4sHHHHHIIIHH',
        SIG_LFH, 20, 0, method, 0, 0,
        crc, len(compressed), len(original),
        len(fname), len(extra)
    ) + fname + extra + compressed

    cd_offset = len(lfh)
    cdh = struct.pack('<4sHHHHHHIIIHHHHHII',
        SIG_CDH, 20, 20, 0, method, 0, 0,
        crc, len(compressed), len(original),
        len(fname), 0, 0, 0, 0, 0, 0
    ) + fname

    eocd = struct.pack('<4sHHHHIIH',
        SIG_EOCD, 0, 0, 1, 1,
        len(cdh), cd_offset, 0
    )

    with open(fpath, 'wb') as f:
        f.write(lfh + cdh + eocd)


def _build_multi_entry_zip(num_entries: int) -> bytes:
    """Build a valid ZIP with multiple benign entries."""
    local_data = b""
    cd_data = b""

    for i in range(num_entries):
        content = f"content of file {i} ".encode() * 20
        compressed = compress_deflate(content)
        crc = zlib.crc32(content) & 0xFFFFFFFF
        fname = f"file_{i:04d}.txt".encode()
        offset = len(local_data)

        lfh = struct.pack('<4sHHHHHIIIHH',
            SIG_LFH, 20, 0, 8, 0, 0,
            crc, len(compressed), len(content),
            len(fname), 0
        ) + fname + compressed

        cdh = struct.pack('<4sHHHHHHIIIHHHHHII',
            SIG_CDH, 20, 20, 0, 8, 0, 0,
            crc, len(compressed), len(content),
            len(fname), 0, 0, 0, 0, 0, offset
        ) + fname

        local_data += lfh
        cd_data += cdh

    cd_offset = len(local_data)
    eocd = struct.pack('<4sHHHHIIH',
        SIG_EOCD, 0, 0,
        num_entries, num_entries,
        len(cd_data), cd_offset, 0
    )

    return local_data + cd_data + eocd


# -- Step 5: Verify the hard test set --
def verify_hard_test():
    """
    Check that eocd_count alone cannot separate the classes.
    This is the key validation - if it can, the test is too easy.
    """
    ev_rows, nev_rows = [], []

    for fname in os.listdir(HARD_EV_DIR):
        if fname.endswith('.zip'):
            f = extract_features(
                os.path.join(HARD_EV_DIR, fname)
            )
            ev_rows.append(f)

    for fname in os.listdir(HARD_NEV_DIR):
        if fname.endswith('.zip'):
            f = extract_features(
                os.path.join(HARD_NEV_DIR, fname)
            )
            nev_rows.append(f)

    ev_df = pd.DataFrame(ev_rows)
    nev_df = pd.DataFrame(nev_rows)

    print(f"\n-- Hard Test Set Verification --------------------")
    print(f"Evasion samples    : {len(ev_df)}")
    print(f"Non-evasion samples: {len(nev_df)}")
    print(f"\nFeature separation check:")
    print(f"{'Feature':<30} {'Evasion':>10} {'Non-evasion':>12}")
    print(f"{'-' * 55}")

    for col in ['eocd_count', 'method_mismatch',
                'data_entropy_shannon',
                'suspicious_entry_ratio',
                'any_crc_mismatch']:
        if col in ev_df.columns:
            print(f"{col:<30} "
                  f"{ev_df[col].mean():>10.4f} "
                  f"{nev_df[col].mean():>12.4f}")

    eocd_ev = ev_df['eocd_count'].mean()
    eocd_nev = nev_df['eocd_count'].mean()
    ratio = eocd_ev / max(eocd_nev, 0.01)

    print(f"\nEOCD ratio (evasion/non-evasion): {ratio:.2f}x")
    if ratio > 10:
        print("WARNING: EOCD still dominates - test may still be too easy")
    elif ratio < 3:
        print("GOOD: EOCD ratio is low - models must use multiple features")
    else:
        print("ACCEPTABLE: Moderate EOCD signal - other features needed too")

    return len(ev_df), len(nev_df)


# -- Main --
if __name__ == "__main__":
    print("Building hard evaluation test set...")
    print("=" * 50)

    n_real_ev = copy_real_evasion()
    n_synth_ev = copy_synthetic_hard_variants(count=22)
    n_real_nev = copy_real_non_evasion()
    n_hard_neg = generate_hard_negatives(count=50)

    print(f"\n{'=' * 50}")
    print(f"Hard test set composition:")
    print(f"  Real evasion       : {n_real_ev}")
    print(f"  Synthetic evasion  : {n_synth_ev}")
    print(f"  Real non-evasion   : {n_real_nev}")
    print(f"  Hard neg benign    : {n_hard_neg}")
    print(f"  Total evasion      : {n_real_ev + n_synth_ev}")
    print(f"  Total non-evasion  : {n_real_nev + n_hard_neg}")
    print(f"  Grand total        : "
          f"{n_real_ev + n_synth_ev + n_real_nev + n_hard_neg}")

    n_ev, n_nev = verify_hard_test()

    print(f"\nSaved to: {HARD_TEST_DIR}")
    print(f"Next: python src/evaluate_hard_test.py")
