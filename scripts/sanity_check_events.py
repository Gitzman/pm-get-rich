"""Sanity check normalized event data.

Per event prints: n_events, n_distinct_wallets, time_range,
median_inter_event_gap, histogram of events per wallet.

Usage:
    uv run python scripts/sanity_check_events.py [--dir data/events]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def check_event(event_dir: Path) -> None:
    """Run sanity checks on a single normalized event directory."""
    meta_path = event_dir / "_meta.json"
    parquet_path = event_dir / "events.parquet"

    if not parquet_path.exists():
        print(f"  SKIP: no events.parquet in {event_dir}")
        return

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    df = pl.read_parquet(parquet_path)
    n_events = df.height

    # Distinct wallets (actors + counterparties)
    actors = df["actor"].unique()
    counterparties = df["counterparty"].unique()
    all_wallets = pl.concat([actors, counterparties]).unique()
    n_wallets = all_wallets.len()

    # Time range
    ts_min = df["timestamp_ms"].min()
    ts_max = df["timestamp_ms"].max()
    duration_hours = (ts_max - ts_min) / (1000 * 3600) if ts_min and ts_max else 0

    # Median inter-event gap (ms)
    sorted_ts = df.sort("seq")["timestamp_ms"]
    gaps = sorted_ts.diff().drop_nulls()
    median_gap_ms = gaps.median() if gaps.len() > 0 else 0
    median_gap_s = (median_gap_ms or 0) / 1000

    # Events per wallet histogram
    actor_counts = df.group_by("actor").len().rename({"len": "n_trades"})
    cp_counts = df.group_by("counterparty").len().rename({
        "counterparty": "actor", "len": "n_trades"
    })
    wallet_trades = (
        pl.concat([actor_counts, cp_counts])
        .group_by("actor")
        .agg(pl.col("n_trades").sum())
    )

    # Histogram buckets
    counts = wallet_trades["n_trades"]
    buckets = [1, 5, 10, 50, 100, 500, 1000]
    hist_parts = []
    for i, threshold in enumerate(buckets):
        lower = buckets[i - 1] if i > 0 else 0
        n = counts.filter((counts > lower) & (counts <= threshold)).len()
        if n > 0:
            hist_parts.append(f"{lower+1}-{threshold}: {n}")
    above = counts.filter(counts > buckets[-1]).len()
    if above > 0:
        hist_parts.append(f">{buckets[-1]}: {above}")

    event_id = meta.get("event_id", event_dir.name)
    title = meta.get("event_title", "")[:50]
    outcome = meta.get("outcome_bucket", "?")

    print(f"\n{'='*70}")
    print(f"Event: {event_id} — {title}")
    print(f"  n_trades:      {n_events:,}")
    print(f"  n_wallets:     {n_wallets:,}")
    print(f"  time_range:    {duration_hours:.1f} hours")
    print(f"  median_gap:    {median_gap_s:.1f}s")
    print(f"  outcome:       {outcome}")
    print(f"  buckets:       {meta.get('n_buckets', '?')} — {meta.get('bucket_labels', [])}")
    print(f"  wallet histogram: {' | '.join(hist_parts)}")

    # Schema validation
    expected_cols = {
        "event_id", "event_title", "seq", "timestamp_ms", "block_number",
        "log_index", "event_type", "actor", "actor_role", "side", "market_id",
        "suit", "price", "size", "usd_amount", "fill_id", "counterparty",
        "counterparty_role", "outcome", "outcome_bucket",
    }
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        print(f"  WARN: missing columns: {missing}")
    if extra:
        print(f"  INFO: extra columns: {extra}")

    # Seq monotonicity check
    seqs = df["seq"]
    if seqs.is_sorted():
        print(f"  seq: monotonic ✓")
    else:
        print(f"  seq: NOT monotonic ✗")

    # Null checks
    null_counts = {
        col: df[col].null_count() for col in df.columns if df[col].null_count() > 0
    }
    if null_counts:
        print(f"  nulls: {null_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check normalized events")
    parser.add_argument(
        "--dir", type=Path, default=Path("data/events"), help="Events directory"
    )
    parser.add_argument("--event", type=str, help="Check single event ID")
    args = parser.parse_args()

    if args.event:
        event_dir = args.dir / args.event
        if not event_dir.exists():
            print(f"Event directory not found: {event_dir}")
            return
        check_event(event_dir)
    else:
        event_dirs = sorted(
            [d for d in args.dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not event_dirs:
            print(f"No event directories found in {args.dir}")
            return
        print(f"Checking {len(event_dirs)} events in {args.dir}")
        for event_dir in event_dirs:
            check_event(event_dir)

    print(f"\n{'='*70}")
    print("Sanity check complete.")


if __name__ == "__main__":
    main()
