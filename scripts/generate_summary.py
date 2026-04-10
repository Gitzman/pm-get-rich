"""Generate aggregated summary.json from per-market generalization results.

Usage:
    uv run python scripts/generate_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

MARKETS = [
    {"event_id": "295980", "city": "Seoul", "date": "March 26"},
    {"event_id": "302794", "city": "Seoul", "date": "March 28"},
    {"event_id": "299412", "city": "Seoul", "date": "March 27"},
    {"event_id": "285764", "city": "Seoul", "date": "March 23"},
    {"event_id": "289192", "city": "Seoul", "date": "March 24"},
    {"event_id": "306323", "city": "Seoul", "date": "March 29"},
    {"event_id": "292561", "city": "Seoul", "date": "March 25"},
    {"event_id": "282546", "city": "NYC", "date": "March 22"},
    {"event_id": "313358", "city": "NYC", "date": "March 31"},
    {"event_id": "289197", "city": "NYC", "date": "March 24"},
    {"event_id": "265850", "city": "NYC", "date": "March 16"},
    {"event_id": "285756", "city": "London", "date": "March 23"},
    {"event_id": "306301", "city": "London", "date": "March 29"},
    {"event_id": "268916", "city": "London", "date": "March 18"},
    {"event_id": "289183", "city": "London", "date": "March 24"},
    {"event_id": "282558", "city": "Shanghai", "date": "March 22"},
    {"event_id": "306340", "city": "Shanghai", "date": "March 29"},
    {"event_id": "292578", "city": "Shanghai", "date": "March 25"},
    {"event_id": "15031", "city": "global", "date": "December 2024"},
    {"event_id": "193611", "city": "global", "date": "February 2026"},
    {"event_id": "16712", "city": "global", "date": "January 2025"},
    {"event_id": "295989", "city": "Chicago", "date": "March 26"},
    {"event_id": "306331", "city": "Miami", "date": "March 29"},
    {"event_id": "262796", "city": "Atlanta", "date": "March 14"},
    {"event_id": "275646", "city": "Toronto", "date": "March 20"},
]

BASE_DIR = Path("data/reports/generalization")


def main() -> None:
    results = []

    for market in MARKETS:
        eid = market["event_id"]
        market_dir = BASE_DIR / eid

        entry = {
            "event_id": eid,
            "city": market["city"],
            "date": market["date"],
            "classical": {},
            "neural": {},
            "status": "pending",
        }

        # Load classical results (prefer classical_results.json, fallback to results.json)
        for classical_candidate in [market_dir / "classical_results.json", market_dir / "results.json"]:
            if classical_candidate.exists():
                with open(classical_candidate) as f:
                    d = json.load(f)
                if d.get("model") == "multivariate_exponential_hawkes":
                    entry["classical"] = {
                        "held_out_avg_log_likelihood": d["metrics"]["held_out_avg_log_likelihood"],
                        "branching_ratio": d["fitted_parameters"]["branching_ratio"],
                        "dimensions": d["dimensions"],
                        "n_train": d["n_train"],
                        "n_test": d["n_test"],
                        "beta": d["fitted_parameters"]["beta"],
                    }
                    entry["status"] = "classical_only"
                    break

        # Load neural results
        neural_path = market_dir / "neural_results.json"
        if neural_path.exists():
            with open(neural_path) as f:
                d = json.load(f)
            if d.get("model") == "neural_hawkes_ct_lstm":
                entry["neural"] = {
                    "held_out_avg_log_likelihood": d["metrics"]["held_out_avg_log_likelihood"],
                    "held_out_ll_std": d["metrics"].get("held_out_ll_std"),
                    "beats_classical": d["metrics"].get("beats_baseline"),
                    "dimensions": d["dimensions"],
                    "n_parameters": d.get("n_parameters"),
                    "training_epochs": d.get("training", {}).get("epochs"),
                    "training_time_seconds": d.get("training", {}).get("training_time_seconds"),
                }
                entry["status"] = "success" if entry.get("classical") else "neural_only"

        results.append(entry)

    # Aggregate
    classical_lls = [r["classical"]["held_out_avg_log_likelihood"]
                     for r in results if r["classical"].get("held_out_avg_log_likelihood") is not None]
    neural_lls = [r["neural"]["held_out_avg_log_likelihood"]
                  for r in results if r["neural"].get("held_out_avg_log_likelihood") is not None]

    neural_beats = sum(1 for r in results
                       if r["neural"].get("held_out_avg_log_likelihood") is not None
                       and r["classical"].get("held_out_avg_log_likelihood") is not None
                       and r["neural"]["held_out_avg_log_likelihood"] > r["classical"]["held_out_avg_log_likelihood"])

    n_both = sum(1 for r in results
                 if r["neural"].get("held_out_avg_log_likelihood") is not None
                 and r["classical"].get("held_out_avg_log_likelihood") is not None)

    # Per-city breakdown
    cities = sorted(set(m["city"] for m in MARKETS))
    city_stats = {}
    for city in cities:
        city_results = [r for r in results if r["city"] == city]
        city_classical = [r["classical"]["held_out_avg_log_likelihood"]
                          for r in city_results if r["classical"].get("held_out_avg_log_likelihood") is not None]
        city_neural = [r["neural"]["held_out_avg_log_likelihood"]
                       for r in city_results if r["neural"].get("held_out_avg_log_likelihood") is not None]
        city_stats[city] = {
            "n_markets": len(city_results),
            "classical_mean_ll": float(sum(city_classical) / len(city_classical)) if city_classical else None,
            "neural_mean_ll": float(sum(city_neural) / len(city_neural)) if city_neural else None,
            "n_classical": len(city_classical),
            "n_neural": len(city_neural),
        }

    summary = {
        "n_markets_total": len(results),
        "n_classical_fitted": len(classical_lls),
        "n_neural_fitted": len(neural_lls),
        "cities": cities,
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
            "comparison": {
                "n_both_fitted": n_both,
                "neural_beats_classical": neural_beats,
                "neural_beats_pct": float(100 * neural_beats / n_both) if n_both else None,
            },
        },
        "per_city": city_stats,
        "per_market": results,
    }

    output_path = BASE_DIR / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {output_path}")
    print(f"  Markets: {len(results)} total")
    print(f"  Classical: {len(classical_lls)} fitted, mean LL = {sum(classical_lls)/len(classical_lls):.4f}" if classical_lls else "  Classical: 0 fitted")
    print(f"  Neural: {len(neural_lls)} fitted, mean LL = {sum(neural_lls)/len(neural_lls):.4f}" if neural_lls else "  Neural: 0 fitted")
    if n_both:
        print(f"  Neural beats classical: {neural_beats}/{n_both} ({100*neural_beats/n_both:.0f}%)")
    print(f"\n  Per-city classical LL:")
    for city, stats in sorted(city_stats.items()):
        ll = stats["classical_mean_ll"]
        n = stats["n_classical"]
        if ll is not None:
            print(f"    {city:>12}: {ll:>8.4f} (n={n})")


if __name__ == "__main__":
    main()
