"""Batch Hawkes fitting pipeline for multi-market generalization study.

Normalizes events, fits classical + neural Hawkes on each market, and
produces per-market reports + aggregated summary.

Usage:
    uv run python scripts/batch_hawkes.py [--gpu 0] [--gpu2 1] [--classical-only]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Market selection: 25 markets across 7 cities + global
# ---------------------------------------------------------------------------

MARKETS = [
    # Seoul (7 consecutive dates - highest volume city)
    {"event_id": "295980", "city": "Seoul", "date": "March 26"},
    {"event_id": "302794", "city": "Seoul", "date": "March 28"},
    {"event_id": "299412", "city": "Seoul", "date": "March 27"},
    {"event_id": "285764", "city": "Seoul", "date": "March 23"},
    {"event_id": "289192", "city": "Seoul", "date": "March 24"},
    {"event_id": "306323", "city": "Seoul", "date": "March 29"},
    {"event_id": "292561", "city": "Seoul", "date": "March 25"},
    # NYC (4 dates)
    {"event_id": "282546", "city": "NYC", "date": "March 22"},
    {"event_id": "313358", "city": "NYC", "date": "March 31"},
    {"event_id": "289197", "city": "NYC", "date": "March 24"},
    {"event_id": "265850", "city": "NYC", "date": "March 16"},
    # London (4 dates)
    {"event_id": "285756", "city": "London", "date": "March 23"},
    {"event_id": "306301", "city": "London", "date": "March 29"},
    {"event_id": "268916", "city": "London", "date": "March 18"},
    {"event_id": "289183", "city": "London", "date": "March 24"},
    # Shanghai (3 dates)
    {"event_id": "282558", "city": "Shanghai", "date": "March 22"},
    {"event_id": "306340", "city": "Shanghai", "date": "March 29"},
    {"event_id": "292578", "city": "Shanghai", "date": "March 25"},
    # Global temperature increase (3 dates)
    {"event_id": "15031", "city": "global", "date": "December 2024"},
    {"event_id": "193611", "city": "global", "date": "February 2026"},
    {"event_id": "16712", "city": "global", "date": "January 2025"},
    # Mid-tier diversity cities
    {"event_id": "295989", "city": "Chicago", "date": "March 26"},
    {"event_id": "306331", "city": "Miami", "date": "March 29"},
    {"event_id": "262796", "city": "Atlanta", "date": "March 14"},
    {"event_id": "275646", "city": "Toronto", "date": "March 20"},
]


def run_cmd(cmd: list[str], label: str, timeout: int = 3600) -> tuple[bool, str]:
    """Run a subprocess, return (success, output)."""
    print(f"  > {' '.join(cmd[:6])}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"    FAILED ({label}): {output[-500:]}")
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT ({label})")
        return False, "timeout"
    except Exception as e:
        print(f"    ERROR ({label}): {e}")
        return False, str(e)


def normalize_market(event_id: str, db_path: str) -> bool:
    """Normalize a single market if not already done."""
    event_dir = Path(f"data/events/{event_id}")
    if (event_dir / "events.parquet").exists():
        print(f"  Already normalized: {event_id}")
        return True

    ok, _ = run_cmd(
        ["uv", "run", "python", "scripts/normalize_events.py",
         "--db", db_path, "--event", event_id],
        f"normalize {event_id}",
    )
    return ok


def fit_classical(event_id: str, out_dir: Path) -> tuple[bool, dict | None]:
    """Fit classical Hawkes on a single market."""
    parquet = Path(f"data/events/{event_id}/events.parquet")
    if not parquet.exists():
        return False, {"error": "no normalized data"}

    results_path = out_dir / "results.json"
    ok, output = run_cmd(
        ["uv", "run", "python", "scripts/fit_hawkes_classical.py",
         "--parquet", str(parquet),
         "--out", str(out_dir),
         "--max-dims", "50",
         "--min-events", "10",
         "--top-wallets", "50",
         "--l1", "0.05"],
        f"classical {event_id}",
        timeout=600,
    )

    if ok and results_path.exists():
        with open(results_path) as f:
            return True, json.load(f)
    return False, {"error": output[-300:] if output else "unknown"}


def fit_neural(event_id: str, out_dir: Path, gpu: int = 0) -> tuple[bool, dict | None]:
    """Fit neural Hawkes on a single market."""
    parquet = Path(f"data/events/{event_id}/events.parquet")
    if not parquet.exists():
        return False, {"error": "no normalized data"}

    results_path = out_dir / "results.json"
    ok, output = run_cmd(
        ["uv", "run", "python", "scripts/fit_hawkes_neural.py",
         "--parquet", str(parquet),
         "--out", str(out_dir),
         "--hidden", "64",
         "--embed-dim", "32",
         "--epochs", "400",
         "--patience", "30",
         "--gpu", str(gpu),
         "--mc-samples", "20",
         "--chunk-size", "512"],
        f"neural {event_id}",
        timeout=3600,
    )

    if ok and results_path.exists():
        with open(results_path) as f:
            return True, json.load(f)
    return False, {"error": output[-300:] if output else "unknown"}


def write_summary(results: list[dict], output_path: Path) -> None:
    """Write aggregated summary.json."""
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    # Aggregate metrics
    classical_lls = [r["classical"]["held_out_avg_log_likelihood"]
                     for r in successful if r.get("classical", {}).get("held_out_avg_log_likelihood") is not None]
    neural_lls = [r["neural"]["held_out_avg_log_likelihood"]
                  for r in successful if r.get("neural", {}).get("held_out_avg_log_likelihood") is not None]

    neural_beats = sum(1 for r in successful
                       if r.get("neural", {}).get("held_out_avg_log_likelihood") is not None
                       and r.get("classical", {}).get("held_out_avg_log_likelihood") is not None
                       and r["neural"]["held_out_avg_log_likelihood"] > r["classical"]["held_out_avg_log_likelihood"])

    summary = {
        "n_markets_total": len(results),
        "n_markets_successful": len(successful),
        "n_markets_failed": len(failed),
        "cities": sorted(set(r["city"] for r in results)),
        "aggregate_metrics": {
            "classical": {
                "n_fitted": len(classical_lls),
                "mean_held_out_ll": float(sum(classical_lls) / len(classical_lls)) if classical_lls else None,
                "min_held_out_ll": float(min(classical_lls)) if classical_lls else None,
                "max_held_out_ll": float(max(classical_lls)) if classical_lls else None,
            },
            "neural": {
                "n_fitted": len(neural_lls),
                "mean_held_out_ll": float(sum(neural_lls) / len(neural_lls)) if neural_lls else None,
                "min_held_out_ll": float(min(neural_lls)) if neural_lls else None,
                "max_held_out_ll": float(max(neural_lls)) if neural_lls else None,
            },
            "neural_beats_classical": neural_beats,
            "neural_beats_pct": float(100 * neural_beats / len(classical_lls)) if classical_lls else None,
        },
        "per_market": results,
        "failed_markets": [{"event_id": r["event_id"], "city": r["city"],
                            "date": r["date"], "reason": r.get("error", "unknown")}
                           for r in failed],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Hawkes fitting pipeline")
    parser.add_argument("--gpu", type=int, default=0, help="Primary GPU for neural fitting")
    parser.add_argument("--db", type=str, default="data/pmgetrich.duckdb", help="DuckDB path")
    parser.add_argument("--classical-only", action="store_true", help="Skip neural fitting")
    parser.add_argument("--neural-only", action="store_true", help="Skip classical fitting (use existing)")
    parser.add_argument("--start-at", type=int, default=0, help="Start at market index N (for resume)")
    parser.add_argument("--max-gpu-hours", type=float, default=4.0, help="Max GPU hours before stopping neural")
    args = parser.parse_args()

    print("=" * 70)
    print("BATCH HAWKES FITTING PIPELINE")
    print(f"Markets: {len(MARKETS)} | GPU: {args.gpu} | Max GPU hours: {args.max_gpu_hours}")
    print("=" * 70)

    base_dir = Path("data/reports/generalization")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Normalize all markets
    print(f"\n{'='*70}")
    print("PHASE 1: NORMALIZING EVENTS")
    print(f"{'='*70}")
    for i, market in enumerate(MARKETS):
        eid = market["event_id"]
        print(f"\n[{i+1}/{len(MARKETS)}] {market['city']} {market['date']} (event={eid})")
        if not normalize_market(eid, args.db):
            print(f"  WARNING: Failed to normalize {eid}, will skip in fitting")

    # Phase 2: Fit classical Hawkes on all markets
    results: list[dict] = []
    gpu_time_total = 0.0

    if not args.neural_only:
        print(f"\n{'='*70}")
        print("PHASE 2: CLASSICAL HAWKES FITTING")
        print(f"{'='*70}")

        for i, market in enumerate(MARKETS):
            if i < args.start_at:
                continue
            eid = market["event_id"]
            out_dir = base_dir / eid
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{i+1}/{len(MARKETS)}] Classical: {market['city']} {market['date']} (event={eid})")
            t0 = time.time()
            ok, result_data = fit_classical(eid, out_dir)
            elapsed = time.time() - t0

            entry = {
                "event_id": eid,
                "city": market["city"],
                "date": market["date"],
                "status": "success" if ok else "failed",
                "classical": {
                    "held_out_avg_log_likelihood": result_data.get("metrics", {}).get("held_out_avg_log_likelihood") if ok else None,
                    "branching_ratio": result_data.get("fitted_parameters", {}).get("branching_ratio") if ok else None,
                    "dimensions": result_data.get("dimensions") if ok else None,
                    "n_train": result_data.get("n_train") if ok else None,
                    "n_test": result_data.get("n_test") if ok else None,
                    "train_nll_per_event": result_data.get("metrics", {}).get("train_nll_per_event") if ok else None,
                    "beta": result_data.get("fitted_parameters", {}).get("beta") if ok else None,
                    "elapsed_seconds": elapsed,
                },
                "neural": {},
            }
            if not ok:
                entry["error"] = str(result_data)
            results.append(entry)

            if ok:
                print(f"    LL={result_data['metrics']['held_out_avg_log_likelihood']:.4f}  "
                      f"BR={result_data['fitted_parameters']['branching_ratio']:.4f}  "
                      f"D={result_data['dimensions']}  {elapsed:.1f}s")
    else:
        # Load existing classical results
        for market in MARKETS:
            eid = market["event_id"]
            classical_results_path = base_dir / eid / "results.json"
            entry = {
                "event_id": eid,
                "city": market["city"],
                "date": market["date"],
                "status": "success",
                "classical": {},
                "neural": {},
            }
            if classical_results_path.exists():
                with open(classical_results_path) as f:
                    result_data = json.load(f)
                entry["classical"] = {
                    "held_out_avg_log_likelihood": result_data.get("metrics", {}).get("held_out_avg_log_likelihood"),
                    "branching_ratio": result_data.get("fitted_parameters", {}).get("branching_ratio"),
                    "dimensions": result_data.get("dimensions"),
                    "n_train": result_data.get("n_train"),
                    "n_test": result_data.get("n_test"),
                    "train_nll_per_event": result_data.get("metrics", {}).get("train_nll_per_event"),
                    "beta": result_data.get("fitted_parameters", {}).get("beta"),
                }
            results.append(entry)

    # Phase 3: Fit neural Hawkes
    if not args.classical_only:
        print(f"\n{'='*70}")
        print("PHASE 3: NEURAL HAWKES FITTING")
        print(f"{'='*70}")

        gpu_budget_seconds = args.max_gpu_hours * 3600

        for i, market in enumerate(MARKETS):
            if i < args.start_at:
                continue
            eid = market["event_id"]
            out_dir = base_dir / eid
            out_dir.mkdir(parents=True, exist_ok=True)

            # Check GPU budget
            if gpu_time_total >= gpu_budget_seconds:
                print(f"\n  GPU budget exhausted ({gpu_time_total/3600:.1f}h >= {args.max_gpu_hours}h). Stopping neural fitting.")
                for j in range(i, len(MARKETS)):
                    if j < len(results):
                        results[j]["neural"] = {"skipped": "gpu_budget_exhausted"}
                    else:
                        results.append({
                            "event_id": MARKETS[j]["event_id"],
                            "city": MARKETS[j]["city"],
                            "date": MARKETS[j]["date"],
                            "status": "skipped",
                            "classical": {},
                            "neural": {"skipped": "gpu_budget_exhausted"},
                        })
                break

            print(f"\n[{i+1}/{len(MARKETS)}] Neural: {market['city']} {market['date']} (event={eid})  "
                  f"[GPU time: {gpu_time_total/60:.0f}m / {gpu_budget_seconds/60:.0f}m]")
            t0 = time.time()
            ok, result_data = fit_neural(eid, out_dir, gpu=args.gpu)
            elapsed = time.time() - t0
            gpu_time_total += elapsed

            # Update the result entry
            if i < len(results):
                entry = results[i]
            else:
                entry = {
                    "event_id": eid,
                    "city": market["city"],
                    "date": market["date"],
                    "status": "success",
                    "classical": {},
                    "neural": {},
                }
                results.append(entry)

            if ok:
                entry["neural"] = {
                    "held_out_avg_log_likelihood": result_data.get("metrics", {}).get("held_out_avg_log_likelihood"),
                    "held_out_ll_std": result_data.get("metrics", {}).get("held_out_ll_std"),
                    "beats_classical": result_data.get("metrics", {}).get("beats_baseline"),
                    "dimensions": result_data.get("dimensions"),
                    "n_parameters": result_data.get("n_parameters"),
                    "training_epochs": result_data.get("training", {}).get("epochs"),
                    "training_time_seconds": result_data.get("training", {}).get("training_time_seconds"),
                    "elapsed_seconds": elapsed,
                }
                neural_ll = result_data.get("metrics", {}).get("held_out_avg_log_likelihood", 0)
                classical_ll = entry.get("classical", {}).get("held_out_avg_log_likelihood")
                beats = "YES" if classical_ll and neural_ll > classical_ll else "NO"
                print(f"    LL={neural_ll:.4f}  beats_classical={beats}  {elapsed:.1f}s")
            else:
                entry["neural"] = {"error": str(result_data)}
                if entry["status"] == "success":
                    entry["status"] = "partial"
                print(f"    FAILED: {elapsed:.1f}s")

            # Write partial summary after every 10 markets
            if (i + 1) % 10 == 0 or i == len(MARKETS) - 1:
                write_summary(results, base_dir / "summary.json")
                print(f"\n  --- Partial summary saved ({i+1}/{len(MARKETS)} markets) ---")

    # Final summary
    write_summary(results, base_dir / "summary.json")

    print(f"\n{'='*70}")
    print("BATCH FITTING COMPLETE")
    print(f"{'='*70}")

    successful = [r for r in results if r.get("status") in ("success", "partial")]
    print(f"  Markets: {len(results)} total, {len(successful)} successful")
    print(f"  GPU time: {gpu_time_total/60:.1f} minutes")
    print(f"  Reports: {base_dir}/")
    print(f"  Summary: {base_dir}/summary.json")


if __name__ == "__main__":
    main()
