"""Run neural Hawkes fits using both GPUs in parallel."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# All 25 markets
ALL_MARKETS = [
    "295980", "302794", "299412", "285764", "289192", "306323", "292561",  # Seoul
    "282546", "313358", "289197", "265850",  # NYC
    "285756", "306301", "268916", "289183",  # London
    "282558", "306340", "292578",  # Shanghai
    "15031", "193611", "16712",  # Global
    "295989", "306331", "262796", "275646",  # Chicago, Miami, Atlanta, Toronto
]


def fit_one(event_id: str, gpu: int) -> tuple[str, bool, float, str]:
    parquet = f"data/events/{event_id}/events.parquet"
    out = f"data/reports/generalization/{event_id}"
    results_path = Path(f"{out}/results.json")

    # Skip if neural results already exist
    neural_results_path = Path(f"{out}/neural_results.json")
    if neural_results_path.exists():
        with open(neural_results_path) as f:
            d = json.load(f)
        ll = d.get("metrics", {}).get("held_out_avg_log_likelihood", 0)
        return event_id, True, 0.0, f"already done (LL={ll:.4f})"

    # Check if normalized data exists
    if not Path(parquet).exists():
        return event_id, False, 0.0, "no normalized data"

    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "python", "scripts/fit_hawkes_neural.py",
         "--parquet", parquet, "--out", out,
         "--hidden", "64", "--embed-dim", "32",
         "--epochs", "400", "--patience", "30",
         "--gpu", str(gpu), "--mc-samples", "20", "--chunk-size", "512"],
        capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - t0

    if result.returncode == 0 and results_path.exists():
        with open(results_path) as f:
            d = json.load(f)
        # Save as neural_results.json to avoid overwriting classical
        neural_path = Path(f"{out}/neural_results.json")
        with open(neural_path, "w") as f:
            json.dump(d, f, indent=2)
        # Restore classical_results.json as results.json if it exists
        classical_path = Path(f"{out}/classical_results.json")
        if classical_path.exists():
            import shutil
            shutil.copy2(str(classical_path), str(results_path))
        ll = d.get("metrics", {}).get("held_out_avg_log_likelihood", 0)
        return event_id, True, elapsed, f"LL={ll:.4f} ({elapsed:.0f}s)"
    return event_id, False, elapsed, result.stderr[-200:] if result.stderr else "unknown error"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gpu-hours", type=float, default=4.0)
    args = parser.parse_args()

    # Filter to markets with normalized data
    markets = [m for m in ALL_MARKETS if Path(f"data/events/{m}/events.parquet").exists()]
    print(f"Neural Hawkes: {len(markets)} markets, 2 GPUs, max {args.max_gpu_hours}h")

    # Assign markets to GPUs in round-robin
    gpu_assignments = [(m, i % 2) for i, m in enumerate(markets)]

    total_gpu_time = 0.0
    budget = args.max_gpu_hours * 3600

    # Run 2 at a time (one per GPU)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        idx = 0
        active = 0

        # Submit first 2
        while idx < len(gpu_assignments) and active < 2:
            eid, gpu = gpu_assignments[idx]
            futures[pool.submit(fit_one, eid, gpu)] = (eid, gpu)
            idx += 1
            active += 1

        while futures:
            done_futures = []
            for future in as_completed(futures):
                eid, gpu = futures[future]
                try:
                    event_id, ok, elapsed, msg = future.result()
                    status = "OK" if ok else "FAIL"
                    total_gpu_time += elapsed
                    remaining_budget = budget - total_gpu_time
                    print(f"  [{status}] {event_id} (GPU {gpu}): {msg}  "
                          f"[total GPU: {total_gpu_time/60:.0f}m, budget: {remaining_budget/60:.0f}m left]")
                except Exception as e:
                    print(f"  [ERROR] {eid}: {e}")

                done_futures.append(future)

                # Submit next if budget allows
                if idx < len(gpu_assignments) and total_gpu_time < budget:
                    next_eid, next_gpu = gpu_assignments[idx]
                    futures[pool.submit(fit_one, next_eid, next_gpu)] = (next_eid, next_gpu)
                    idx += 1
                elif total_gpu_time >= budget:
                    print(f"\n  GPU budget exhausted ({total_gpu_time/3600:.1f}h). Stopping.")

            for f in done_futures:
                del futures[f]

    print(f"\nTotal GPU time: {total_gpu_time/60:.1f} min ({total_gpu_time/3600:.2f} h)")
    print(f"Markets fitted: {sum(1 for m in markets if Path(f'data/reports/generalization/{m}/results.json').exists())}/{len(markets)}")


if __name__ == "__main__":
    main()
