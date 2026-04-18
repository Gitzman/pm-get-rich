"""Re-run backtest with L2 order-book fill simulator (pm-gy0.4).

Apples-to-apples comparison: same signals, same exit rules, only the fill
model changes. Swaps the statistical fill model (fills.py) for the L2
order-book simulator (book_fills.py) using actual PMXT snapshots.

Input:
  - data/signals/signals.parquet (5,532 TPP signals — NOT re-generated)
  - PMXT L2 order book data from r2.pmxt.dev (March 25-31 2026)

Output:
  - data/backtest/l2_fill_results.parquet (per-signal L2 fill results)
  - data/backtest/l2_comparison.md (side-by-side markdown table)
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from src.costs.book_fills import L2FillSimulator, FillResult
from src.costs.fees import taker_fee
from src.costs.fills import (
    FillAssumptions,
    DEFAULT_FILL_ASSUMPTIONS,
    fill_probability,
    adverse_selection_cost,
)
from src.ingest.pmxt_loader import (
    BookSnapshot,
    _download_parquet,
    _parquet_filename,
    parse_book_snapshot,
    PMXT_CACHE_DIR,
)

# ---------------------------------------------------------------------------
# Suit extraction (from normalize_events.py)
# ---------------------------------------------------------------------------
_CITY_TEMP_EXACT_C = re.compile(r"be (\d+)°C on")
_CITY_TEMP_LOW_C = re.compile(r"be (\d+)°C or below")
_CITY_TEMP_HIGH_C = re.compile(r"be (\d+)°C or higher")
_CITY_TEMP_RANGE_F = re.compile(r"between (\d+)-(\d+)°F on")
_CITY_TEMP_LOW_F = re.compile(r"be (\d+)°F or below")
_CITY_TEMP_HIGH_F = re.compile(r"be (\d+)°F or higher")


def extract_suit(question: str) -> str:
    """Extract bucket label (suit) from a market question."""
    m = _CITY_TEMP_LOW_C.search(question)
    if m:
        return f"<={m.group(1)}C"
    m = _CITY_TEMP_HIGH_C.search(question)
    if m:
        return f">={m.group(1)}C"
    m = _CITY_TEMP_EXACT_C.search(question)
    if m:
        return f"{m.group(1)}C"
    m = _CITY_TEMP_LOW_F.search(question)
    if m:
        return f"<={m.group(1)}F"
    m = _CITY_TEMP_HIGH_F.search(question)
    if m:
        return f">={m.group(1)}F"
    m = _CITY_TEMP_RANGE_F.search(question)
    if m:
        return f"{m.group(1)}-{m.group(2)}F"
    return question


# ---------------------------------------------------------------------------
# Fill regimes (same as backtest_signals.py for comparison)
# ---------------------------------------------------------------------------
FILL_REGIMES: dict[str, dict] = {
    "conservative": {
        "taker_volume": 50,
        "resting_depth": 200,
        "adverse_ticks": 1.0,
        "queue_frac": 1.0,
    },
    "moderate": {
        "taker_volume": 200,
        "resting_depth": 100,
        "adverse_ticks": 0.5,
        "queue_frac": 1.0,
    },
    "optimistic": {
        "taker_volume": 500,
        "resting_depth": 50,
        "adverse_ticks": 0.25,
        "queue_frac": 0.5,
    },
}
CONTRACTS = 100


# ---------------------------------------------------------------------------
# Gamma API: event_id → condition_id mapping
# ---------------------------------------------------------------------------
def build_condition_id_map(
    event_ids: list[str],
) -> dict[tuple[str, str], str]:
    """Map (event_id, suit) → conditionId via Gamma API.

    Returns:
        Dict mapping (event_id, suit_label) to conditionId hex string.
    """
    mapping: dict[tuple[str, str], str] = {}
    failed: list[str] = []

    with httpx.Client(timeout=30.0) as client:
        for i, eid in enumerate(event_ids):
            url = f"https://gamma-api.polymarket.com/events/{eid}"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  WARN: Failed to fetch event {eid}: {e}", flush=True)
                failed.append(eid)
                continue

            markets = data.get("markets", [])
            for m in markets:
                cid = m.get("conditionId", "")
                question = m.get("question", "")
                if cid and question:
                    suit = extract_suit(question)
                    mapping[(eid, suit)] = cid

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(event_ids)}] mapped {len(mapping)} markets", flush=True)

    if failed:
        print(f"  WARNING: {len(failed)} events failed to fetch", flush=True)

    return mapping


# ---------------------------------------------------------------------------
# L2 fill simulation for a batch of signals
# ---------------------------------------------------------------------------
def simulate_l2_fills_for_hour(
    signals_in_hour: pl.DataFrame,
    condition_map: dict[tuple[str, str], str],
    hour_dt: datetime,
    assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
) -> list[dict]:
    """Simulate L2 fills for all signals in a given hour.

    Downloads the PMXT hour file, extracts snapshots for needed condition_ids,
    and runs the L2 simulator on each signal.

    Returns list of per-signal result dicts.
    """
    # Determine which condition_ids we need for this batch
    needed_cids: set[str] = set()
    for row in signals_in_hour.iter_rows(named=True):
        key = (row["event_id"], row["current_suit"])
        cid = condition_map.get(key)
        if cid:
            needed_cids.add(cid)

    if not needed_cids:
        # No mapped condition_ids — return unfilled results
        return [
            {
                "filled": False,
                "fill_time_s": None,
                "queue_pos_initial": 0.0,
                "snapshots_observed": 0,
                "condition_id": None,
            }
            for _ in range(signals_in_hour.height)
        ]

    # Download hour file and next hour (for signals near the boundary)
    snapshots: list[BookSnapshot] = []
    for dt_offset in [timedelta(hours=0), timedelta(hours=1)]:
        dt = hour_dt + dt_offset
        try:
            parquet_path = _download_parquet(dt)
        except Exception:
            continue

        df = pl.read_parquet(str(parquet_path))
        filtered = df.filter(
            (pl.col("market_id").is_in(list(needed_cids)))
            & (pl.col("update_type") == "book_snapshot")
        )
        for row in filtered.iter_rows(named=True):
            snap = parse_book_snapshot(row)
            if snap is not None:
                snapshots.append(snap)

    if not snapshots:
        return [
            {
                "filled": False,
                "fill_time_s": None,
                "queue_pos_initial": 0.0,
                "snapshots_observed": 0,
                "condition_id": None,
            }
            for _ in range(signals_in_hour.height)
        ]

    # Build simulator
    sim = L2FillSimulator(snapshots=snapshots)

    # Simulate each signal
    results = []
    for row in signals_in_hour.iter_rows(named=True):
        key = (row["event_id"], row["current_suit"])
        cid = condition_map.get(key)

        if not cid or cid not in sim._by_market:
            results.append({
                "filled": False,
                "fill_time_s": None,
                "queue_pos_initial": 0.0,
                "snapshots_observed": 0,
                "condition_id": cid,
            })
            continue

        # Determine order parameters
        price = row["current_price"]
        pred_bucket_pos = row["pred_bucket_pos"]
        side = "buy" if pred_bucket_pos > 0.5 else "sell"
        place_time = datetime.fromtimestamp(
            row["timestamp_ms"] / 1000, tz=timezone.utc
        )

        fill_result = sim.simulate_order(
            market_id=cid,
            price=price,
            size=CONTRACTS,
            side=side,
            place_time=place_time,
            assumptions=assumptions,
        )

        results.append({
            "filled": fill_result.filled,
            "fill_time_s": fill_result.time_to_fill_s,
            "queue_pos_initial": fill_result.queue_position_initial,
            "snapshots_observed": fill_result.snapshots_observed,
            "condition_id": cid,
        })

    return results


# ---------------------------------------------------------------------------
# Old statistical fill model (for comparison)
# ---------------------------------------------------------------------------
def compute_old_fill_results(
    df: pl.DataFrame,
    regime_name: str,
) -> dict:
    """Run the old statistical fill model on all signals.

    Returns aggregate metrics matching backtest_signals.py output format.
    """
    regime = FILL_REGIMES[regime_name]
    assumptions = FillAssumptions(
        adverse_selection_ticks=regime["adverse_ticks"],
        queue_position_frac=regime["queue_frac"],
    )
    taker_vol = regime["taker_volume"]
    resting_depth = regime["resting_depth"]

    prices = df["current_price"].to_numpy()
    price_changes = df["price_change"].to_numpy()
    pred_bucket_pos = df["pred_bucket_pos"].to_numpy()
    directions = np.where(pred_bucket_pos > 0.5, 1.0, -1.0)
    n = len(prices)

    fill_probs = np.zeros(n)
    gross_pnl = np.zeros(n)
    exit_fees = np.zeros(n)
    adv_sel_costs = np.zeros(n)
    net_pnl = np.zeros(n)

    for i in range(n):
        d = directions[i]
        p = prices[i]
        pc = price_changes[i]

        gross_pnl[i] = d * pc * CONTRACTS
        fill_probs[i] = fill_probability(
            taker_volume_at_price=taker_vol,
            resting_depth_at_price=resting_depth,
            our_size=CONTRACTS,
            assumptions=assumptions,
        )
        exit_price = np.clip(p + pc, 0.01, 0.99)
        exit_fees[i] = taker_fee(float(exit_price), CONTRACTS, 0.050)
        adv_sel_costs[i] = adverse_selection_cost(CONTRACTS, assumptions=assumptions)
        net_pnl[i] = gross_pnl[i] - exit_fees[i] - adv_sel_costs[i]

    expected_pnl = fill_probs * net_pnl
    correct_dir = np.sum((directions * price_changes) > 0)
    flat = np.sum(np.abs(price_changes) < 1e-6)
    hit_rate = float(correct_dir) / max(n - int(flat), 1)
    win_rate = float((expected_pnl > 0).sum()) / n if n > 0 else 0.0

    return {
        "n_signals": n,
        "n_fills": float(fill_probs.sum()),
        "fill_rate": float(fill_probs.mean()),
        "mean_pnl_per_fill": float(net_pnl.mean()),
        "total_pnl": float((fill_probs * net_pnl).sum()),
        "win_rate": win_rate,
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# Compute L2 fill metrics
# ---------------------------------------------------------------------------
def compute_l2_fill_metrics(
    df: pl.DataFrame,
    l2_results: list[dict],
) -> dict:
    """Compute metrics from L2 fill simulation results."""
    prices = df["current_price"].to_numpy()
    price_changes = df["price_change"].to_numpy()
    pred_bucket_pos = df["pred_bucket_pos"].to_numpy()
    directions = np.where(pred_bucket_pos > 0.5, 1.0, -1.0)
    n = len(prices)

    filled = np.array([r["filled"] for r in l2_results], dtype=bool)
    n_fills = int(filled.sum())
    # "Covered" = signal had L2 book data (at least one snapshot observed).
    # Distinguishes PMXT data absence from real unfilled orders.
    covered = np.array(
        [r.get("snapshots_observed", 0) > 0 for r in l2_results], dtype=bool
    )
    n_covered = int(covered.sum())

    # For filled trades, compute P&L
    gross_pnl = directions * price_changes * CONTRACTS
    exit_fees_arr = np.zeros(n)
    adv_sel_arr = np.zeros(n)

    for i in range(n):
        exit_price = np.clip(prices[i] + price_changes[i], 0.01, 0.99)
        exit_fees_arr[i] = taker_fee(float(exit_price), CONTRACTS, 0.050)
        # Use default adverse selection (same as old model's moderate regime)
        adv_sel_arr[i] = adverse_selection_cost(
            CONTRACTS, assumptions=DEFAULT_FILL_ASSUMPTIONS
        )

    net_pnl = gross_pnl - exit_fees_arr - adv_sel_arr

    # Only count P&L for filled trades
    filled_pnl = net_pnl[filled]
    total_pnl = float(filled_pnl.sum()) if n_fills > 0 else 0.0
    mean_pnl = float(filled_pnl.mean()) if n_fills > 0 else 0.0

    # Bootstrap 95% CI on mean P&L per filled trade
    ci_low, ci_high = 0.0, 0.0
    if n_fills >= 10:
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(10000):
            sample = rng.choice(filled_pnl, size=n_fills, replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

    # Direction accuracy for filled trades
    filled_dirs = directions[filled]
    filled_pc = price_changes[filled]
    if n_fills > 0:
        correct = np.sum((filled_dirs * filled_pc) > 0)
        flat = np.sum(np.abs(filled_pc) < 1e-6)
        hit_rate = float(correct) / max(n_fills - int(flat), 1)
        win_rate = float((filled_pnl > 0).sum()) / n_fills
    else:
        hit_rate = 0.0
        win_rate = 0.0

    # Adverse selection drift (use price_change as proxy at different windows)
    # We have signals at dt_seconds = 15, 30, 60, 120
    drift_30s = drift_60s = drift_120s = 0.0
    if n_fills > 0:
        # Use actual price changes from signals at different dt_seconds
        filled_df = df.filter(pl.Series(filled))
        for dt_s, field_name in [(30, "drift_30s"), (60, "drift_60s"), (120, "drift_120s")]:
            dt_subset = filled_df.filter(pl.col("dt_seconds") == dt_s)
            if dt_subset.height > 0:
                if field_name == "drift_30s":
                    drift_30s = float(dt_subset["price_change"].mean())
                elif field_name == "drift_60s":
                    drift_60s = float(dt_subset["price_change"].mean())
                elif field_name == "drift_120s":
                    drift_120s = float(dt_subset["price_change"].mean())

    return {
        "n_signals": n,
        "n_covered": n_covered,
        "coverage_rate": n_covered / n if n > 0 else 0.0,
        "n_fills": n_fills,
        "fill_rate": n_fills / n if n > 0 else 0.0,
        "fill_rate_covered": n_fills / n_covered if n_covered > 0 else 0.0,
        "mean_pnl_per_fill": mean_pnl,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "hit_rate": hit_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "drift_30s": drift_30s,
        "drift_60s": drift_60s,
        "drift_120s": drift_120s,
    }


# ---------------------------------------------------------------------------
# Per-market stratification
# ---------------------------------------------------------------------------
def stratify_by_market(
    df: pl.DataFrame,
    l2_results: list[dict],
) -> pl.DataFrame:
    """Compute fill rate and metrics stratified by event_id.

    Distinguishes covered (had L2 snapshots) from uncovered signals so the
    bottom of the table isn't ambiguous about data absence vs genuine 0% fill.
    """
    filled_col = [r["filled"] for r in l2_results]
    covered_col = [r.get("snapshots_observed", 0) > 0 for r in l2_results]
    df_with_fills = df.with_columns(
        pl.Series("l2_filled", filled_col),
        pl.Series("l2_covered", covered_col),
    )

    market_stats = df_with_fills.group_by("event_id", "city").agg(
        pl.len().alias("n_signals"),
        pl.col("l2_covered").sum().alias("n_covered"),
        pl.col("l2_filled").sum().alias("n_fills"),
        pl.col("l2_filled").mean().alias("fill_rate"),
        pl.col("l2_covered").mean().alias("coverage_rate"),
    ).sort("fill_rate", descending=True)

    return market_stats


# ---------------------------------------------------------------------------
# Adverse selection measurement
# ---------------------------------------------------------------------------
def compute_adverse_selection_drift(
    df: pl.DataFrame,
    l2_results: list[dict],
) -> dict[str, float]:
    """Compute post-fill adverse selection drift at 30s, 60s, 120s.

    Uses price_change column grouped by dt_seconds for filled signals.
    For filled signals, price_change IS the drift at that dt horizon.
    """
    filled = np.array([r["filled"] for r in l2_results], dtype=bool)
    pred_bucket_pos = df["pred_bucket_pos"].to_numpy()
    directions = np.where(pred_bucket_pos > 0.5, 1.0, -1.0)

    result = {}
    for dt_s in [30, 60, 120]:
        dt_mask = (df["dt_seconds"] == dt_s).to_numpy()
        combined = filled & dt_mask
        if combined.sum() == 0:
            result[f"adv_sel_{dt_s}s"] = 0.0
            continue
        # Adverse selection = mean drift AGAINST our direction
        pc = df["price_change"].to_numpy()[combined]
        dirs = directions[combined]
        # Positive drift * BUY direction = favorable; negative = adverse
        # Adverse selection = -1 * mean(direction * price_change)
        # If we bought and price went down, that's adverse
        signed_drift = dirs * pc
        result[f"adv_sel_{dt_s}s"] = float(-signed_drift.mean())

    return result


# ---------------------------------------------------------------------------
# Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(
    old_results: dict[str, dict],
    l2_metrics: dict,
    adv_sel: dict[str, float],
    market_stats: pl.DataFrame,
) -> str:
    """Format the side-by-side comparison as markdown."""
    lines = []
    lines.append("# L2 Fill Simulator Backtest Comparison (pm-gy0.8)")
    lines.append("")
    lines.append("**Apples-to-apples**: same 5,532 TPP signals, same exit rules.")
    lines.append("**Only change**: fill model swapped from statistical → L2 order book.")
    lines.append("")

    # Coverage (data availability, NOT fill outcome)
    n_sig = l2_metrics["n_signals"]
    n_cov = l2_metrics["n_covered"]
    cov_rate = l2_metrics["coverage_rate"]
    n_fills = l2_metrics["n_fills"]
    fill_rate_cov = l2_metrics["fill_rate_covered"]
    lines.append("## Coverage (PMXT L2 data availability)")
    lines.append("")
    lines.append(
        f"- **Signals with L2 book data:** {n_cov:,} / {n_sig:,} = "
        f"**{cov_rate*100:.1f}%**"
    )
    lines.append(
        f"- **Signals without L2 data:** {n_sig - n_cov:,} "
        f"(PMXT hourly parquets missing for those hours — not unfilled orders)"
    )
    lines.append(
        f"- **Fills on covered signals:** {n_fills:,} / {n_cov:,} = "
        f"**{fill_rate_cov*100:.1f}%** (true L2 fill rate when data is present)"
    )
    lines.append("")
    lines.append(
        "All L2 metrics below are computed on the covered subset. "
        "Uncovered signals cannot be evaluated and are excluded from P&L, CI, hit rate, "
        "etc. (they are not \"unfilled\" — the data is simply absent)."
    )
    lines.append("")

    # Headline: does per-trade P&L CI exclude zero?
    ci_low = l2_metrics["ci_low"]
    ci_high = l2_metrics["ci_high"]
    excludes_zero = ci_low > 0 or ci_high < 0
    sign_hint = ""
    if excludes_zero:
        sign_hint = " (NEGATIVE)" if ci_high < 0 else " (POSITIVE)"
    ci_verdict = ("YES" + sign_hint) if excludes_zero else "NO"
    lines.append(f"## Headline: Per-trade P&L 95% CI excludes zero? **{ci_verdict}**")
    lines.append(f"  - CI: [{ci_low:.4f}, {ci_high:.4f}]")
    lines.append(f"  - Mean P&L per filled trade: {l2_metrics['mean_pnl_per_fill']:.4f}")
    lines.append(f"  - Based on {n_fills:,} filled trades drawn from {n_cov:,} covered signals ({cov_rate*100:.1f}% of 5,532).")
    lines.append("")

    # Side-by-side table: old regimes vs L2
    lines.append("## Side-by-Side: Old Fill Model vs L2 Simulator")
    lines.append("")
    lines.append("| Metric | Conservative | Moderate | Optimistic | **L2 Book** |")
    lines.append("|--------|-------------|----------|------------|-------------|")

    metrics_keys = [
        ("n_signals", "Signals", "{:.0f}"),
        ("n_fills", "Fills", "{:.1f}"),
        ("fill_rate", "Fill Rate (of all)", "{:.3f}"),
        ("mean_pnl_per_fill", "Mean P&L/Fill", "{:.4f}"),
        ("total_pnl", "Total P&L", "{:.2f}"),
        ("win_rate", "Win Rate", "{:.3f}"),
        ("hit_rate", "Hit Rate", "{:.3f}"),
    ]

    for key, label, fmt in metrics_keys:
        vals = []
        for regime in ["conservative", "moderate", "optimistic"]:
            v = old_results[regime].get(key, 0)
            vals.append(fmt.format(v))
        l2_v = fmt.format(l2_metrics.get(key, 0))
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | **{l2_v}** |")

    # Coverage-aware L2 rows
    lines.append(
        f"| Signals w/ L2 data | - | - | - | **{l2_metrics['n_covered']}** |"
    )
    lines.append(
        f"| Fill Rate (of covered) | - | - | - | "
        f"**{l2_metrics['fill_rate_covered']:.3f}** |"
    )
    lines.append(f"| P&L 95% CI | - | - | - | **[{ci_low:.4f}, {ci_high:.4f}]** |")
    lines.append("")

    # Adverse selection
    lines.append("## Adverse Selection Drift (L2 Fills Only)")
    lines.append("")
    lines.append("| Window | Mean Adverse Drift |")
    lines.append("|--------|-------------------|")
    for dt_s in [30, 60, 120]:
        v = adv_sel.get(f"adv_sel_{dt_s}s", 0.0)
        lines.append(f"| {dt_s}s | {v:.6f} |")
    lines.append("")

    # Market stratification — covered events only (excluding data gaps)
    lines.append("## Fill Rate by Market")
    lines.append("")
    covered_events = market_stats.filter(pl.col("n_covered") > 0)
    uncovered_events = market_stats.filter(pl.col("n_covered") == 0)
    lines.append(
        f"_{covered_events.height} events had L2 coverage; "
        f"{uncovered_events.height} events had NO PMXT data (listed separately)._"
    )
    lines.append("")

    lines.append("### Top 10 Covered Events by Fill Rate")
    lines.append("| Event | City | Signals | Covered | Fills | Fill Rate (of covered) |")
    lines.append("|-------|------|---------|---------|-------|------------------------|")
    for row in covered_events.head(10).iter_rows(named=True):
        rate = row["n_fills"] / row["n_covered"] if row["n_covered"] else 0.0
        lines.append(
            f"| {row['event_id']} | {row['city']} | {row['n_signals']} "
            f"| {row['n_covered']} | {row['n_fills']} | {rate:.3f} |"
        )
    lines.append("")

    lines.append("### Bottom 10 Covered Events by Fill Rate")
    lines.append("| Event | City | Signals | Covered | Fills | Fill Rate (of covered) |")
    lines.append("|-------|------|---------|---------|-------|------------------------|")
    for row in covered_events.sort("fill_rate").head(10).iter_rows(named=True):
        rate = row["n_fills"] / row["n_covered"] if row["n_covered"] else 0.0
        lines.append(
            f"| {row['event_id']} | {row['city']} | {row['n_signals']} "
            f"| {row['n_covered']} | {row['n_fills']} | {rate:.3f} |"
        )
    lines.append("")

    lines.append(
        f"### Events with NO PMXT Coverage ({uncovered_events.height})"
    )
    lines.append("_These had zero L2 snapshots — data absence, not 0% fill._")
    lines.append("")
    if uncovered_events.height > 0:
        lines.append("| Event | City | Signals |")
        lines.append("|-------|------|---------|")
        for row in uncovered_events.sort("city").head(25).iter_rows(named=True):
            lines.append(
                f"| {row['event_id']} | {row['city']} | {row['n_signals']} |"
            )
        if uncovered_events.height > 25:
            lines.append(f"| _(+{uncovered_events.height - 25} more)_ | | |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Re-run backtest with L2 fill simulator (pm-gy0.4)"
    )
    parser.add_argument(
        "--signals", type=Path, default=Path("data/signals/signals.parquet"),
    )
    parser.add_argument(
        "--out-results", type=Path,
        default=Path("data/backtest/l2_fill_results.parquet"),
    )
    parser.add_argument(
        "--out-comparison", type=Path,
        default=Path("data/backtest/l2_comparison.md"),
    )
    parser.add_argument(
        "--cache-map", type=Path,
        default=Path("data/backtest/condition_id_map.json"),
        help="Cache file for Gamma API condition_id mapping",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip PMXT download (use cached files only)",
    )
    args = parser.parse_args()

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Load signals
    # -----------------------------------------------------------------------
    print("Step 1: Loading signals...", flush=True)
    df = pl.read_parquet(args.signals)
    print(f"  {len(df):,} signals, {df['event_id'].n_unique()} events", flush=True)
    ts_min_ms = int(df["timestamp_ms"].min())
    ts_max_ms = int(df["timestamp_ms"].max())
    ts_min_dt = datetime.fromtimestamp(ts_min_ms / 1000, tz=timezone.utc)
    ts_max_dt = datetime.fromtimestamp(ts_max_ms / 1000, tz=timezone.utc)
    print(
        f"  Time range: {ts_min_dt.isoformat()} (epoch_ms={ts_min_ms}) "
        f"to {ts_max_dt.isoformat()} (epoch_ms={ts_max_ms})",
        flush=True,
    )
    if ts_min_dt.year != 2026 or ts_max_dt.year != 2026:
        raise RuntimeError(
            f"TIMESTAMP SANITY FAIL: signals span {ts_min_dt.year}-{ts_max_dt.year}, "
            f"expected 2026. Refusing to run — fix signals.parquet first."
        )

    # -----------------------------------------------------------------------
    # Step 2: Build event_id+suit → conditionId mapping
    # -----------------------------------------------------------------------
    print("\nStep 2: Building condition_id mapping...", flush=True)
    args.cache_map.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_map.exists():
        print(f"  Loading cached mapping from {args.cache_map}", flush=True)
        raw_map = json.loads(args.cache_map.read_text())
        condition_map = {tuple(k.split("|")): v for k, v in raw_map.items()}
    else:
        event_ids = sorted(df["event_id"].unique().to_list())
        print(f"  Fetching {len(event_ids)} events from Gamma API...", flush=True)
        condition_map = build_condition_id_map(event_ids)
        # Cache
        raw_map = {f"{k[0]}|{k[1]}": v for k, v in condition_map.items()}
        args.cache_map.write_text(json.dumps(raw_map, indent=2))
        print(f"  Cached mapping to {args.cache_map}", flush=True)

    print(f"  Mapped {len(condition_map)} (event_id, suit) → conditionId pairs", flush=True)

    # Check coverage
    needed_pairs = set(
        zip(df["event_id"].to_list(), df["current_suit"].to_list())
    )
    mapped_pairs = needed_pairs & set(condition_map.keys())
    unmapped = needed_pairs - mapped_pairs
    print(f"  Coverage: {len(mapped_pairs)}/{len(needed_pairs)} unique pairs mapped", flush=True)
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped pairs (signals will be marked unfilled)", flush=True)
        for p in list(unmapped)[:5]:
            print(f"    {p}", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Determine hours to download
    # -----------------------------------------------------------------------
    print("\nStep 3: Organizing signals by hour...", flush=True)
    ts_ms_arr = df["timestamp_ms"].to_numpy()
    hour_indices: dict[datetime, list[int]] = defaultdict(list)

    for i, ts in enumerate(ts_ms_arr):
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        hour = dt.replace(minute=0, second=0, microsecond=0)
        hour_indices[hour].append(i)

    hours_sorted = sorted(hour_indices.keys())
    print(f"  {len(hours_sorted)} unique hours, first={hours_sorted[0]}, last={hours_sorted[-1]}", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Process signals hour by hour with L2 simulator
    # -----------------------------------------------------------------------
    print("\nStep 4: Running L2 fill simulation...", flush=True)
    all_l2_results: list[dict | None] = [None] * len(df)
    total_filled = 0
    total_processed = 0
    total_no_data = 0

    for hi, hour_dt in enumerate(hours_sorted):
        indices = hour_indices[hour_dt]
        n_in_hour = len(indices)

        # Determine needed condition_ids for this hour
        needed_cids: set[str] = set()
        idx_to_cid: dict[int, str | None] = {}
        for idx in indices:
            row = df.row(idx, named=True)
            key = (row["event_id"], row["current_suit"])
            cid = condition_map.get(key)
            idx_to_cid[idx] = cid
            if cid:
                needed_cids.add(cid)

        if not needed_cids:
            for idx in indices:
                all_l2_results[idx] = {
                    "filled": False,
                    "fill_time_s": None,
                    "queue_pos_initial": 0.0,
                    "snapshots_observed": 0,
                    "condition_id": idx_to_cid.get(idx),
                }
            total_processed += n_in_hour
            total_no_data += n_in_hour
            continue

        # Load hour file(s) from cache and extract snapshots using lazy scan
        # Only read needed columns + rows for memory efficiency
        cid_list = list(needed_cids)
        snapshots: list[BookSnapshot] = []
        for dt_offset in [timedelta(hours=0), timedelta(hours=1)]:
            dt = hour_dt + dt_offset
            # Use cached file if available; download only if --download flag set
            parquet_path = PMXT_CACHE_DIR / _parquet_filename(dt)
            if not parquet_path.exists():
                if not args.skip_download:
                    try:
                        parquet_path = _download_parquet(dt)
                    except Exception as exc:
                        print(
                            f"  WARN: download failed for {dt.isoformat()} "
                            f"(epoch={int(dt.timestamp())}): {exc}",
                            flush=True,
                        )
                        continue
                else:
                    continue

            # Use scan_parquet with predicate pushdown for efficiency
            filtered = (
                pl.scan_parquet(str(parquet_path))
                .filter(
                    (pl.col("market_id").is_in(cid_list))
                    & (pl.col("update_type") == "book_snapshot")
                )
                .collect()
            )
            for row in filtered.iter_rows(named=True):
                snap = parse_book_snapshot(row)
                if snap is not None:
                    snapshots.append(snap)
            del filtered
            gc.collect()

        # Build simulator
        sim = L2FillSimulator(snapshots=snapshots) if snapshots else None

        # Process each signal
        for idx in indices:
            row = df.row(idx, named=True)
            cid = idx_to_cid[idx]

            if not cid or sim is None or cid not in sim._by_market:
                all_l2_results[idx] = {
                    "filled": False,
                    "fill_time_s": None,
                    "queue_pos_initial": 0.0,
                    "snapshots_observed": 0,
                    "condition_id": cid,
                }
                total_no_data += 1
                continue

            price = row["current_price"]
            pred_bucket_pos = row["pred_bucket_pos"]
            side = "buy" if pred_bucket_pos > 0.5 else "sell"
            place_time = datetime.fromtimestamp(
                row["timestamp_ms"] / 1000, tz=timezone.utc
            )

            fill_result = sim.simulate_order(
                market_id=cid,
                price=price,
                size=CONTRACTS,
                side=side,
                place_time=place_time,
            )

            all_l2_results[idx] = {
                "filled": fill_result.filled,
                "fill_time_s": fill_result.time_to_fill_s,
                "queue_pos_initial": fill_result.queue_position_initial,
                "snapshots_observed": fill_result.snapshots_observed,
                "condition_id": cid,
            }
            if fill_result.filled:
                total_filled += 1

        total_processed += n_in_hour
        del snapshots, sim
        gc.collect()

        if (hi + 1) % 5 == 0 or (hi + 1) == len(hours_sorted):
            elapsed = time.time() - t_start
            print(
                f"  [{hi+1}/{len(hours_sorted)}] "
                f"processed={total_processed:,}, "
                f"filled={total_filled:,}, "
                f"no_data={total_no_data:,}, "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Verify all results populated
    assert all(r is not None for r in all_l2_results), "Some signals not processed"

    # -----------------------------------------------------------------------
    # Step 5: Save per-signal L2 results
    # -----------------------------------------------------------------------
    print("\nStep 5: Saving per-signal results...", flush=True)
    results_df = pl.DataFrame({
        "event_id": df["event_id"],
        "city": df["city"],
        "timestamp_ms": df["timestamp_ms"],
        "dt_seconds": df["dt_seconds"],
        "threshold_pct": df["threshold_pct"],
        "current_price": df["current_price"],
        "current_suit": df["current_suit"],
        "pred_bucket_pos": df["pred_bucket_pos"],
        "price_change": df["price_change"],
        "direction": pl.Series(
            np.where(df["pred_bucket_pos"].to_numpy() > 0.5, "buy", "sell")
        ),
        "l2_filled": pl.Series([r["filled"] for r in all_l2_results]),
        "l2_fill_time_s": pl.Series(
            [r["fill_time_s"] for r in all_l2_results],
            dtype=pl.Float64,
        ),
        "l2_queue_pos_initial": pl.Series(
            [r["queue_pos_initial"] for r in all_l2_results],
            dtype=pl.Float64,
        ),
        "l2_snapshots_observed": pl.Series(
            [r["snapshots_observed"] for r in all_l2_results],
            dtype=pl.Int64,
        ),
        "condition_id": pl.Series(
            [r["condition_id"] for r in all_l2_results],
            dtype=pl.Utf8,
        ),
    })

    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(args.out_results)
    print(f"  Saved: {args.out_results} ({len(results_df)} rows)", flush=True)

    # -----------------------------------------------------------------------
    # Step 6: Compute comparison metrics
    # -----------------------------------------------------------------------
    print("\nStep 6: Computing comparison metrics...", flush=True)

    # Old model metrics (for each fill regime)
    old_results = {}
    for regime in FILL_REGIMES:
        old_results[regime] = compute_old_fill_results(df, regime)
        print(f"  Old {regime}: fills={old_results[regime]['n_fills']:.1f}, "
              f"fill_rate={old_results[regime]['fill_rate']:.3f}, "
              f"total_pnl={old_results[regime]['total_pnl']:.2f}")

    # L2 metrics
    l2_metrics = compute_l2_fill_metrics(df, all_l2_results)
    print(f"  L2: fills={l2_metrics['n_fills']}, "
          f"fill_rate={l2_metrics['fill_rate']:.3f}, "
          f"total_pnl={l2_metrics['total_pnl']:.2f}, "
          f"CI=[{l2_metrics['ci_low']:.4f}, {l2_metrics['ci_high']:.4f}]")

    # Adverse selection
    adv_sel = compute_adverse_selection_drift(df, all_l2_results)
    for k, v in adv_sel.items():
        print(f"  {k}: {v:.6f}", flush=True)

    # Market stratification
    market_stats = stratify_by_market(df, all_l2_results)

    # -----------------------------------------------------------------------
    # Step 7: Write comparison report
    # -----------------------------------------------------------------------
    print("\nStep 7: Writing comparison report...", flush=True)
    report = format_comparison_table(old_results, l2_metrics, adv_sel, market_stats)
    args.out_comparison.write_text(report)
    print(f"  Saved: {args.out_comparison}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
