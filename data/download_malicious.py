import io
import os
import time
import zipfile

import requests

OUTPUT_DIR = "data/raw/malicious"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
API_KEY_ENV = "MALWAREBAZAAR_API_KEY"


def _get_headers() -> dict:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {API_KEY_ENV} before running."
        )
    return {"Auth-Key": api_key}


def query_by_tag(tag: str, limit: int = 100) -> list:
    payload = {"query": "get_taginfo", "tag": tag, "limit": limit}
    try:
        resp = requests.post(BAZAAR_API, data=payload, headers=_get_headers(), timeout=30)
        result = resp.json()
        print(f"  API status: {result.get('query_status')}")
        return result.get("data", [])
    except Exception as exc:
        print(f"  Query failed: {exc}")
    return []


def query_recent_zip(selector: int = 100) -> list:
    """Get recent samples from MalwareBazaar and filter ZIP after query."""
    payload = {"query": "get_recent", "selector": str(selector)}
    try:
        resp = requests.post(BAZAAR_API, data=payload, headers=_get_headers(), timeout=30)
        result = resp.json()
        print(f"  API status: {result.get('query_status')}")
        return result.get("data", [])
    except Exception as exc:
        print(f"  Query failed: {exc}")
    return []


def download_sample(sha256: str, output_path: str) -> bool:
    payload = {"query": "get_file", "sha256_hash": sha256}
    try:
        resp = requests.post(BAZAAR_API, data=payload, headers=_get_headers(), timeout=60)
        if resp.status_code == 200 and len(resp.content) > 100:
            # MalwareBazaar wraps all downloads in ZIP with password "infected".
            # Some samples may use unsupported ZIP methods in stdlib zipfile.
            try:
                archive = zipfile.ZipFile(io.BytesIO(resp.content))
                for name in archive.namelist():
                    if not name.endswith("/"):
                        data = archive.read(name, pwd=b"infected")
                        with open(output_path, "wb") as file:
                            file.write(data)
                        return True
            except Exception:
                # Fallback: keep wrapped ZIP container for later offline extraction tools.
                with open(output_path, "wb") as file:
                    file.write(resp.content)
                return True
    except Exception as exc:
        print(f"  Download failed {sha256[:12]}: {exc}")
    return False


def download_daily_batch(date_str: str):
    """
    Download an entire day's batch of malware samples.
    date_str format: YYYY-MM-DD  e.g. "2026-03-10"
    """
    url = f"https://bazaar.abuse.ch/export/zip/{date_str}/"
    print(f"\nDownloading daily batch for {date_str}...")
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=120, stream=True)
        if resp.status_code == 200:
            batch_path = os.path.join(OUTPUT_DIR, f"batch_{date_str}.zip")
            with open(batch_path, "wb") as file:
                for chunk in resp.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"  Batch saved: {batch_path}")
            return batch_path
        print(f"  Batch download failed: HTTP {resp.status_code}")
    except Exception as exc:
        print(f"  Batch download error: {exc}")
    return None


if __name__ == "__main__":
    downloaded = 0

    # Method 1: Query by tag (with Auth-Key)
    tags = ["gootloader", "zip", "loader", "archive"]
    for tag in tags:
        print(f"\nQuerying tag: {tag}")
        samples = query_by_tag(tag, limit=100)
        print(f"  Found {len(samples)} samples")

        for sample in samples:
            sha256 = sample.get("sha256_hash", "")
            filetype = sample.get("file_type", "").lower()
            filename = sample.get("file_name", "").lower()

            if "zip" not in filetype and "zip" not in filename:
                continue

            out_path = os.path.join(OUTPUT_DIR, f"bazaar_{sha256[:16]}.zip")
            if os.path.exists(out_path):
                continue

            print(f"  Downloading {sha256[:16]}...")
            if download_sample(sha256, out_path):
                downloaded += 1
                print(f"  OK ({downloaded} total)")

            time.sleep(1)

    # Method 2: Recent samples (broader net)
    print("\nQuerying recent samples...")
    recent = query_recent_zip(selector=100)
    for sample in recent:
        sha256 = sample.get("sha256_hash", "")
        filetype = sample.get("file_type", "").lower()

        if "zip" not in filetype:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"bazaar_recent_{sha256[:16]}.zip")
        if os.path.exists(out_path):
            continue

        print(f"  Downloading {sha256[:16]}...")
        if download_sample(sha256, out_path):
            downloaded += 1
        time.sleep(1)

    print(f"\nTotal downloaded via API: {downloaded}")
    print(f"Total malicious samples: {len(os.listdir(OUTPUT_DIR))}")