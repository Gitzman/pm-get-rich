"""Run classical Hawkes fits in parallel across markets."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REMAINING = [
    "313358", "289197", "265850",  # NYC
    "285756", "306301", "268916", "289183",  # London
    "282558", "306340", "292578",  # Shanghai
    "15031", "193611", "16712",  # Global
    "295989", "306331", "262796", "275646",  # Chicago, Miami, Atlanta, Toronto
]


def fit_one(event_id: str) -> tuple[str, bool, str]:
    parquet = f"data/events/{event_id}/events.parquet"
    out = f"data/reports/generalization/{event_id}"
    if Path(f"{out}/results.json").exists():
        with open(f"{out}/results.json") as f:
            d = json.load(f)
        if d.get("model") == "multivariate_exponential_hawkes":
            return event_id, True, "already done"

    result = subprocess.run(
        ["uv", "run", "python", "scripts/fit_hawkes_classical.py",
         "--parquet", parquet, "--out", out,
         "--max-dims", "50", "--min-events", "10", "--top-wallets", "50", "--l1", "0.05"],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode == 0:
        return event_id, True, "success"
    return event_id, False, result.stderr[-300:]


def main():
    n_workers = 4
    print(f"Running {len(REMAINING)} classical fits with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fit_one, eid): eid for eid in REMAINING}
        for future in as_completed(futures):
            eid = futures[future]
            try:
                event_id, ok, msg = future.result()
                status = "OK" if ok else "FAIL"
                print(f"  [{status}] {event_id}: {msg[:80]}")
            except Exception as e:
                print(f"  [ERROR] {eid}: {e}")


if __name__ == "__main__":
    main()
