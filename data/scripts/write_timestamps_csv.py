"""Rebuild bazaar_timestamps.csv from the progress JSON."""
import csv, json

progress = json.load(open("data/timestamp_progress.json"))
rows = list(progress.values())
rows.sort(key=lambda r: r.get("first_seen") or "")

with open("data/bazaar_timestamps.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sha256_short", "sha256_full", "first_seen", "tags", "file_name", "file_type"])
    w.writeheader()
    w.writerows(rows)

with_ts = sum(1 for r in rows if r.get("first_seen"))
print(f"Written {len(rows)} rows, {with_ts} with timestamps")
dates = sorted(r["first_seen"] for r in rows if r.get("first_seen"))
if dates:
    print(f"Date range: {dates[0]} -> {dates[-1]}")
