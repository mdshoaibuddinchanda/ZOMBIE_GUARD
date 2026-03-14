import requests
import zipfile
import io
import os
import time
from requests.exceptions import RequestException

try:
    import pyzipper
except Exception:
    pyzipper = None

OUTPUT_DIR = "data/real_world_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.environ.get("MALWAREBAZAAR_API_KEY", "")
if not API_KEY:
    raise ValueError("Set MALWAREBAZAAR_API_KEY environment variable first")

HEADERS = {"Auth-Key": API_KEY}
BAZAAR = "https://mb-api.abuse.ch/api/v1/"


def _download_wrapped_sample(sha256: str) -> bytes | None:
    """Download one wrapped sample payload from MalwareBazaar."""
    try:
        dl = requests.post(
            BAZAAR,
            data={"query": "get_file", "sha256_hash": sha256},
            headers=HEADERS,
            timeout=60,
        )
        if dl.status_code != 200 or len(dl.content) < 50:
            return None
        return dl.content
    except RequestException as e:
        print(f"  Network error: {sha256[:16]} - {e}")
        return None


def _extract_wrapped_payload(blob: bytes) -> bytes | None:
    """Extract MalwareBazaar wrapped ZIP content using password 'infected'."""
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
        for name in z.namelist():
            if not name.endswith("/"):
                return z.read(name, pwd=b"infected")
    except Exception:
        pass

    if pyzipper is not None:
        try:
            with pyzipper.AESZipFile(io.BytesIO(blob)) as z:
                z.setpassword(b"infected")
                for name in z.namelist():
                    if not name.endswith("/"):
                        return z.read(name)
        except Exception:
            pass

    return None


def download_by_tag(tag: str, limit: int = 100) -> int:
    """Query by tag and download ZIP-type samples."""
    print(f"\nQuerying tag: {tag}")
    resp = requests.post(
        BAZAAR,
        data={"query": "get_taginfo", "tag": tag, "limit": limit},
        headers=HEADERS,
        timeout=30,
    )
    data = resp.json()
    print(f"  Status: {data.get('query_status')}")
    samples = data.get("data") or []
    print(f"  Found:  {len(samples)} samples")

    saved = 0
    for s in samples:
        sha256 = s.get("sha256_hash", "")
        filetype = s.get("file_type", "").lower()
        fname = s.get("file_name", "").lower()

        # Only keep ZIP files
        if "zip" not in filetype and not fname.endswith(".zip"):
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{sha256[:20]}.zip")
        if os.path.exists(out_path):
            continue

        wrapped = _download_wrapped_sample(sha256)
        if wrapped is None:
            continue

        raw = _extract_wrapped_payload(wrapped)
        if raw is None:
            print(f"  Failed: {sha256[:16]} - unable to unpack wrapped sample")
            continue

        with open(out_path, "wb") as f:
            f.write(raw)
        saved += 1
        print(f"  Saved: {sha256[:16]} ({filetype})")

        time.sleep(0.5)

    return saved


def download_recent(limit: int = 100) -> int:
    """Download most recent submissions, filter for ZIP."""
    print("\nQuerying recent submissions...")
    resp = requests.post(
        BAZAAR,
        data={"query": "get_recent", "selector": str(limit)},
        headers=HEADERS,
        timeout=30,
    )
    data = resp.json()
    samples = data.get("data") or []
    print(f"  Found: {len(samples)} recent samples")

    saved = 0
    for s in samples:
        sha256 = s.get("sha256_hash", "")
        filetype = s.get("file_type", "").lower()

        if "zip" not in filetype:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"recent_{sha256[:20]}.zip")
        if os.path.exists(out_path):
            continue

        wrapped = _download_wrapped_sample(sha256)
        if wrapped is None:
            continue

        raw = _extract_wrapped_payload(wrapped)
        if raw is None:
            print(f"  Failed: {sha256[:16]} - unable to unpack wrapped sample")
            continue

        with open(out_path, "wb") as f:
            f.write(raw)
        saved += 1
        print(f"  Saved recent: {sha256[:16]}")

        time.sleep(0.5)

    return saved


if __name__ == "__main__":
    total = 0

    # Search these tags - all documented ZIP malware families
    for tag in ["gootloader", "zip", "loader", "archive", "dropper"]:
        try:
            total += download_by_tag(tag, limit=100)
        except RequestException as e:
            print(f"Tag query failed for {tag}: {e}")
        except Exception as e:
            print(f"Unexpected failure for tag {tag}: {e}")

    # Also grab recent ZIP submissions
    try:
        total += download_recent(limit=100)
    except RequestException as e:
        print(f"Recent query failed: {e}")
    except Exception as e:
        print(f"Unexpected failure in recent query: {e}")

    files = os.listdir(OUTPUT_DIR)
    print(f"\n{'-' * 50}")
    print(f"Total downloaded : {total}")
    print(f"Files in folder  : {len(files)}")
    print(f"Saved to         : {OUTPUT_DIR}")
    print(f"{'-' * 50}")
    print("\nNEXT STEP:")
    print("python src/detector.py --batch data/real_world_validation/")
