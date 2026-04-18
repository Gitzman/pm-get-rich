"""Pre-download PMXT hourly parquet files in parallel.

Reads signal hours from signals.parquet, determines missing PMXT cache files,
and downloads them concurrently. Each hour file is ~300-600MB.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from src.ingest.pmxt_loader import (
    PMXT_BASE_URL,
    PMXT_CACHE_DIR,
    _parquet_filename,
)


def _download_one(dt: datetime, cache: Path, timeout: float = 300.0) -> tuple[datetime, str]:
    """Download a single PMXT hour file."""
    fname = _parquet_filename(dt)
    out = cache / fname
    if out.exists():
        return dt, "cached"

    url = f"{PMXT_BASE_URL}/{fname}"
    headers = {"User-Agent": "prediction-market-backtesting/1.0"}
    tmp = out.with_suffix(".parquet.tmp")
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
        tmp.rename(out)
        return dt, f"ok ({out.stat().st_size / 1e6:.1f} MB)"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return dt, f"FAIL: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals", type=Path, default=Path("data/signals/signals.parquet")
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", type=Path, default=PMXT_CACHE_DIR)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"Loading signals from {args.signals}...", flush=True)
    df = pl.read_parquet(args.signals)
    ts_ms = df["timestamp_ms"].to_numpy()
    hours: set[datetime] = set()
    for t in ts_ms:
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        hours.add(dt)

    # For each signal hour, we need hour and hour+1 (for boundary signals)
    needed: set[datetime] = set()
    for h in hours:
        needed.add(h)
        needed.add(h + timedelta(hours=1))

    print(f"  {len(hours)} unique signal hours → {len(needed)} parquet files needed", flush=True)

    # Check cache
    missing = sorted(
        h for h in needed if not (args.cache / _parquet_filename(h)).exists()
    )
    print(f"  {len(needed) - len(missing)} cached, {len(missing)} missing", flush=True)

    if not missing:
        print("All files present. Done.")
        return

    # Validate first and last look like 2026
    print(
        f"  Download range: {missing[0].isoformat()} (epoch={int(missing[0].timestamp())}) "
        f"to {missing[-1].isoformat()} (epoch={int(missing[-1].timestamp())})",
        flush=True,
    )
    if missing[0].year != 2026 or missing[-1].year != 2026:
        print("ERROR: dates not in 2026", file=sys.stderr)
        sys.exit(1)

    # Download in parallel
    t0 = time.time()
    n_ok = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_one, dt, args.cache): dt for dt in missing}
        for i, fut in enumerate(as_completed(futs)):
            dt, status = fut.result()
            if status.startswith("ok"):
                n_ok += 1
            elif status != "cached":
                n_fail += 1
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(missing)}] {dt.isoformat()} → {status} "
                f"(elapsed={elapsed:.0f}s, eta={remaining:.0f}s)",
                flush=True,
            )

    print(f"\nDone: {n_ok} downloaded, {n_fail} failed, elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
