"""
download_realworld.py
Aggressive real-world sample collection from MalwareBazaar.
Targets archive evasion malware families across 6 years of data.
Target: 1000-1500 confirmed malicious ZIP samples.
"""

import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests

try:
    import pyzipper
except Exception:
    pyzipper = None

OUTPUT_DIR = "data/real_world_validation"
PROGRESS_FILE = "data/download_progress.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.environ.get("MALWAREBAZAAR_API_KEY", "")
if not API_KEY:
    raise ValueError("Set MALWAREBAZAAR_API_KEY environment variable")

HEADERS = {"Auth-Key": API_KEY}
BAZAAR = "https://mb-api.abuse.ch/api/v1/"

# Families known to use archive evasion techniques.
TARGET_TAGS = [
    "gootloader",
    "gootkit",
    "zip",
    "archive",
    "loader",
    "dropper",
    "jscript",
    "javascript",
    "nanocore",
    "asyncrat",
    "remcos",
    "agentesla",
    "formbook",
    "lokibot",
    "qakbot",
    "emotet",
    "icedid",
    "bazarloader",
]

PRIORITY_TAGS = ["gootloader", "gootkit", "zip", "archive"]


def load_progress() -> set[str]:
    """Load already-downloaded SHA256 hashes."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return set(json.load(file))
    return set()


def save_progress(downloaded: set[str]) -> None:
    """Persist downloaded SHA256 hashes for resumable runs."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(list(downloaded), file)


def query_by_tag(tag: str, limit: int = 1000) -> list[dict]:
    """Query MalwareBazaar for samples by tag."""
    payload = {"query": "get_taginfo", "tag": tag, "limit": limit}
    try:
        response = requests.post(BAZAAR, data=payload, headers=HEADERS, timeout=30)
        result = response.json()
        status = result.get("query_status")
        data = result.get("data") or []
        print(f"  Tag '{tag}': status={status}, found={len(data)}")
        return data
    except Exception as exc:
        print(f"  Tag '{tag}' query failed: {exc}")
        return []


def query_by_filetype(filetype: str = "zip", limit: int = 1000) -> list[dict]:
    """Query MalwareBazaar for samples by file type."""
    payload = {"query": "get_file_type", "file_type": filetype, "limit": limit}
    try:
        response = requests.post(BAZAAR, data=payload, headers=HEADERS, timeout=30)
        result = response.json()
        data = result.get("data") or []
        print(f"  Filetype '{filetype}': found={len(data)}")
        return data
    except Exception as exc:
        print(f"  Filetype query failed: {exc}")
        return []


def query_recent(selector: str = "100") -> list[dict]:
    """Get most recent submissions."""
    payload = {"query": "get_recent", "selector": selector}
    try:
        response = requests.post(BAZAAR, data=payload, headers=HEADERS, timeout=30)
        result = response.json()
        return result.get("data") or []
    except Exception as exc:
        print(f"  Recent query failed: {exc}")
        return []


def _extract_wrapped_payload(blob: bytes) -> tuple[bytes | None, str]:
    """Extract MalwareBazaar wrapped payload using password 'infected'."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            try:
                return archive.read(name, pwd=b"infected"), "ok_zipfile"
            except RuntimeError:
                # Often indicates AES-encrypted wrappers unsupported by stdlib zipfile.
                break
    except Exception:
        pass

    if pyzipper is not None:
        try:
            with pyzipper.AESZipFile(io.BytesIO(blob)) as archive:
                archive.setpassword(b"infected")
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    return archive.read(name), "ok_pyzipper"
        except Exception:
            return None, "extract_failed_pyzipper"

    return None, "extract_failed_no_pyzipper"


def download_sample(sha256: str, output_path: str) -> tuple[bool, str]:
    """Download one sample by SHA256."""
    payload = {"query": "get_file", "sha256_hash": sha256}
    try:
        response = requests.post(BAZAAR, data=payload, headers=HEADERS, timeout=60)
        if response.status_code != 200:
            return False, f"http_{response.status_code}"
        if len(response.content) < 50:
            return False, "response_too_small"

        raw, extract_reason = _extract_wrapped_payload(response.content)
        if raw is None:
            return False, extract_reason

        if not (raw[:2] == b"PK" or len(raw) > 100):
            return False, "payload_filter_reject"

        with open(output_path, "wb") as file:
            file.write(raw)
        return True, extract_reason
    except requests.Timeout:
        return False, "timeout"
    except requests.RequestException:
        return False, "network_error"
    except Exception:
        return False, "unexpected_error"


def is_zip_file(sample: dict) -> bool:
    """Check whether sample metadata indicates an archive-like ZIP payload."""
    filetype = sample.get("file_type", "").lower()
    filename = sample.get("file_name", "").lower()
    tags = [tag.lower() for tag in sample.get("tags", []) or []]

    return (
        "zip" in filetype
        or filename.endswith(".zip")
        or filename.endswith(".jar")
        or filename.endswith(".apk")
        or "zip" in tags
    )


def collect_all_samples(target: int = 1500) -> int:
    """Collect candidates from multiple sources and download until target."""
    downloaded_hashes = load_progress()
    existing_files = len(list(Path(OUTPUT_DIR).glob("*.zip")))
    print(f"Already have: {existing_files} files")
    print(f"Target: {target} files")
    print(f"Need:   {max(0, target - existing_files)} more\n")

    if existing_files >= target:
        print("Target already reached.")
        return existing_files

    all_samples: list[dict] = []

    print("=== Round 1: Priority tags ===")
    for tag in PRIORITY_TAGS:
        samples = query_by_tag(tag, limit=1000)
        all_samples.extend(samples)
        time.sleep(1)

    print("\n=== Round 2: All target tags ===")
    for tag in TARGET_TAGS:
        if tag in PRIORITY_TAGS:
            continue
        samples = query_by_tag(tag, limit=1000)
        all_samples.extend(samples)
        time.sleep(1)

    print("\n=== Round 3: ZIP file type query ===")
    zip_samples = query_by_filetype("zip", limit=1000)
    all_samples.extend(zip_samples)
    time.sleep(1)

    print("\n=== Round 4: Recent submissions ===")
    recent = query_recent("100")
    all_samples.extend([sample for sample in recent if is_zip_file(sample)])

    seen = set()
    unique = []
    for sample in all_samples:
        sha = sample.get("sha256_hash", "")
        if sha and sha not in seen:
            seen.add(sha)
            unique.append(sample)

    print(f"\nTotal unique candidates: {len(unique)}")
    print("Filtering for ZIP files...")

    zip_candidates = [sample for sample in unique if is_zip_file(sample)]
    print(f"ZIP candidates: {len(zip_candidates)}")

    print("\n=== Downloading ===")
    newly_downloaded = 0
    failed = 0
    skipped_known = 0
    checked = 0
    heartbeat_every = 50
    started_at = time.time()
    fail_reasons: dict[str, int] = {}

    for sample in zip_candidates:
        checked += 1
        current_total = existing_files + newly_downloaded
        if current_total >= target:
            print(f"\nTarget of {target} reached.")
            break

        sha256 = sample.get("sha256_hash", "")
        if not sha256 or sha256 in downloaded_hashes:
            skipped_known += 1
            if checked % heartbeat_every == 0:
                elapsed = int(time.time() - started_at)
                print(
                    f"  Heartbeat: checked={checked} downloaded={newly_downloaded} "
                    f"failed={failed} skipped={skipped_known} elapsed={elapsed}s"
                )
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{sha256[:20]}.zip")
        if os.path.exists(out_path):
            downloaded_hashes.add(sha256)
            skipped_known += 1
            if checked % heartbeat_every == 0:
                elapsed = int(time.time() - started_at)
                print(
                    f"  Heartbeat: checked={checked} downloaded={newly_downloaded} "
                    f"failed={failed} skipped={skipped_known} elapsed={elapsed}s"
                )
            continue

        family_tags = sample.get("tags", ["unknown"])
        family = family_tags[0] if family_tags else "unknown"

        success, reason = download_sample(sha256, out_path)

        if success:
            newly_downloaded += 1
            downloaded_hashes.add(sha256)
            total = existing_files + newly_downloaded
            print(f"  [{total:4d}/{target}] {sha256[:16]} ({family})")

            if newly_downloaded % 50 == 0:
                save_progress(downloaded_hashes)
                print(f"  Progress saved. Total: {total} files")
        else:
            failed += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            if failed % 10 == 0:
                print(f"  Failed downloads so far: {failed}")
                top_reasons = sorted(
                    fail_reasons.items(), key=lambda item: item[1], reverse=True
                )[:3]
                reason_line = ", ".join([f"{k}={v}" for k, v in top_reasons])
                if reason_line:
                    print(f"  Top fail reasons: {reason_line}")

        if checked % heartbeat_every == 0:
            elapsed = int(time.time() - started_at)
            print(
                f"  Heartbeat: checked={checked} downloaded={newly_downloaded} "
                f"failed={failed} skipped={skipped_known} elapsed={elapsed}s"
            )

        time.sleep(0.5)

    save_progress(downloaded_hashes)

    final_count = len(list(Path(OUTPUT_DIR).glob("*.zip")))
    print(f"\n{'=' * 50}")
    print(f"Session downloaded : {newly_downloaded}")
    print(f"Failed             : {failed}")
    if fail_reasons:
        print("Fail reasons       :")
        for key, value in sorted(
            fail_reasons.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  - {key}: {value}")
    print(f"Total in folder    : {final_count}")
    print(f"{'=' * 50}")

    return final_count


if __name__ == "__main__":
    print("ZombieGuard - Real Sample Collection")
    print("Target: 1500 confirmed malicious ZIP samples")
    print("Source: MalwareBazaar API")
    print("=" * 50 + "\n")

    final = collect_all_samples(target=1500)

    print(f"\nCollection complete: {final} files")
    print("\nNext steps:")
    print("1. python data/verify_realworld.py")
    print("2. python data/split_realworld.py")
    print("3. python src/classifier_realworld.py")