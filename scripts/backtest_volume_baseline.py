"""Volume baseline backtest (pm-kcw.1).

Simplest volume-timing signal: rolling volume in sliding windows, fire when
volume exceeds Nth percentile. Direction = net BUY/SELL pressure in the window.

Runs through the SAME backtest pipeline as TPP signals (same cost model,
fill regimes, and metrics).

Volume windows: 30s, 60s, 300s (5min)
Look-ahead (dt_seconds): 15s, 30s, 60s, 120s  (same as TPP)
Fill regimes: conservative, moderate, optimistic (same as TPP)

Target: ~485 signals to match TPP trade count at threshold_pct=10.

Output: data/backtest/volume_baseline_results.parquet
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.costs.fees import taker_fee
from src.costs.fills import FillAssumptions, fill_probability, adverse_selection_cost

# ---------------------------------------------------------------------------
# Fill regimes (same as backtest_signals.py)
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

# Volume lookback windows (seconds).
VOLUME_WINDOWS = [30, 60, 300]

# Look-ahead windows for measuring price change (same as TPP).
DT_VALUES = [15, 30, 60, 120]

# Cutoff for held-out events (same as TPP).
CUTOFF_EPOCH = datetime.datetime(2026, 3, 25, tzinfo=datetime.timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Volume signal generation
# ---------------------------------------------------------------------------

def generate_volume_signals_for_event(
    event_id: str,
    events_dir: Path,
    volume_windows: list[int] = VOLUME_WINDOWS,
    dt_values: list[int] = DT_VALUES,
    step_frac: int = 4,
) -> list[dict]:
    """Generate volume-spike signals for one event.

    For each step in the test window (last 20%), computes rolling volume
    (contract size) in each lookback window. Records the volume strength
    (percentile rank within this event) and the net direction of trades.

    Does NOT apply thresholds here — returns all candidate signals with
    their volume_strength score so the caller can apply global thresholds.
    """
    parquet_path = events_dir / event_id / "events.parquet"
    meta_path = events_dir / event_id / "_meta.json"
    if not parquet_path.exists():
        return []

    df = pl.read_parquet(parquet_path).sort("seq")
    N = df.height
    if N < 100:
        return []

    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    city = meta.get("city", "unknown")

    timestamps_ms = df["timestamp_ms"].to_numpy().astype(np.float64)
    prices = df["price"].to_numpy().astype(np.float64)
    sizes = df["size"].to_numpy().astype(np.float64)
    sides = df["side"].to_list()
    suits = df["suit"].to_list()

    # Signed sizes: positive for BUY, negative for SELL
    side_signs = np.array([1.0 if s == "BUY" else -1.0 for s in sides])
    signed_sizes = sizes * side_signs

    # Use last 20% as signal generation window (same as TPP)
    split = int(0.8 * N)
    if N - split < 20:
        return []

    # For each volume window, pre-compute rolling volumes at each step
    # to compute within-event percentiles for thresholding.
    signals = []

    for vol_win_s in volume_windows:
        vol_win_ms = vol_win_s * 1000

        # First pass: compute rolling volume at every step in test window
        # to establish the percentile distribution for this event+window.
        all_volumes = []
        step_size = max(1, (N - split) // (4 * step_frac))

        for i in range(split, N, step_size):
            ts_now = timestamps_ms[i]
            lookback_start = ts_now - vol_win_ms
            # Find trades in [lookback_start, ts_now)
            mask = (timestamps_ms[:i] >= lookback_start)
            vol = float(sizes[:i][mask].sum())
            all_volumes.append(vol)

        if not all_volumes:
            continue

        all_volumes_arr = np.array(all_volumes)

        # Second pass: generate signals at each step point
        step_idx = 0
        for i in range(split, N, step_size):
            ts_now = timestamps_ms[i]
            lookback_start = ts_now - vol_win_ms

            # Trades in the lookback window
            mask = (timestamps_ms[:i] >= lookback_start)
            window_sizes = sizes[:i][mask]
            window_signed = signed_sizes[:i][mask]

            vol = float(window_sizes.sum())
            n_trades = int(mask.sum())

            if n_trades < 2 or vol < 1e-6:
                step_idx += 1
                continue

            # Volume strength = percentile rank within this event
            volume_strength = float((all_volumes_arr <= vol).sum()) / len(all_volumes_arr)

            # Direction: net signed volume (BUY - SELL)
            net_signed = float(window_signed.sum())
            direction = 1.0 if net_signed > 0 else -1.0

            # Encode as pred_bucket_pos equivalent: 0.5 + direction * strength/2
            # This makes it compatible with compute_signal_direction() from backtest_signals
            pred_bucket_pos = 0.5 + direction * volume_strength * 0.5

            current_price = float(prices[i - 1])
            current_suit = suits[i - 1]

            # Look ahead for each dt window
            for dt_s in dt_values:
                dt_ms = dt_s * 1000
                future_mask = (timestamps_ms[i:] - ts_now) <= dt_ms
                n_future = int(future_mask.sum())

                if n_future > 0:
                    future_prices = prices[i:i + n_future]
                    price_at_end = float(future_prices[-1])
                    price_change = price_at_end - current_price
                    max_price = float(future_prices.max())
                    min_price = float(future_prices.min())
                else:
                    price_change = 0.0
                    max_price = current_price
                    min_price = current_price

                signals.append({
                    "event_id": event_id,
                    "city": city,
                    "timestamp_ms": int(ts_now),
                    "volume_window_s": vol_win_s,
                    "dt_seconds": dt_s,
                    "rolling_volume": round(vol, 2),
                    "n_trades_in_window": n_trades,
                    "net_signed_volume": round(net_signed, 2),
                    "volume_strength": round(volume_strength, 4),
                    "pred_bucket_pos": round(pred_bucket_pos, 6),
                    "current_price": round(current_price, 4),
                    "current_suit": current_suit,
                    "price_change": round(price_change, 6),
                    "max_price_in_window": round(max_price, 4),
                    "min_price_in_window": round(min_price, 4),
                    "n_future_events": n_future,
                })

            step_idx += 1

    return signals


# ---------------------------------------------------------------------------
# Backtest (reused from backtest_signals.py)
# ---------------------------------------------------------------------------

def backtest_group(
    df: pl.DataFrame,
    directions: np.ndarray,
    fill_assumptions: FillAssumptions,
    taker_vol: float,
    resting_depth: float,
    contracts: float = CONTRACTS,
    theta: float = 0.050,
) -> dict:
    """Run backtest on a group of signals with given directions and fill params."""
    prices = df["current_price"].to_numpy()
    price_changes = df["price_change"].to_numpy()
    n = len(prices)

    if n == 0:
        return _empty_result()

    gross_pnl = np.zeros(n)
    fill_probs = np.zeros(n)
    exit_fees = np.zeros(n)
    adv_sel_costs = np.zeros(n)
    net_pnl = np.zeros(n)
    expected_pnl = np.zeros(n)

    for i in range(n):
        d = directions[i]
        p = prices[i]
        pc = price_changes[i]

        gross_pnl[i] = d * pc * contracts

        fill_probs[i] = fill_probability(
            taker_volume_at_price=taker_vol,
            resting_depth_at_price=resting_depth,
            our_size=contracts,
            assumptions=fill_assumptions,
        )

        exit_price = np.clip(p + pc, 0.01, 0.99)
        exit_fees[i] = taker_fee(float(exit_price), contracts, theta)

        adv_sel_costs[i] = adverse_selection_cost(
            contracts, assumptions=fill_assumptions
        )

        net_pnl[i] = gross_pnl[i] - exit_fees[i] - adv_sel_costs[i]
        expected_pnl[i] = fill_probs[i] * net_pnl[i]

    n_filled = float(fill_probs.sum())
    total_gross = float(gross_pnl.sum())
    total_fees = float(exit_fees.sum())
    total_adv_sel = float(adv_sel_costs.sum())
    total_net = float(net_pnl.sum())
    total_expected = float(expected_pnl.sum())

    correct_dir = np.sum((directions * price_changes) > 0)
    flat = np.sum(np.abs(price_changes) < 1e-6)
    hit_rate = float(correct_dir) / max(n - int(flat), 1)

    if n > 1 and expected_pnl.std() > 0:
        sharpe = float(expected_pnl.mean() / expected_pnl.std())
    else:
        sharpe = 0.0

    win_rate = float((expected_pnl > 0).sum()) / n

    return {
        "n_signals": n,
        "n_filled_expected": round(n_filled, 2),
        "total_gross_pnl": round(total_gross, 4),
        "total_exit_fees": round(total_fees, 4),
        "total_adverse_selection": round(total_adv_sel, 4),
        "total_net_pnl": round(total_net, 4),
        "total_expected_pnl": round(total_expected, 4),
        "mean_expected_pnl": round(total_expected / n, 6),
        "hit_rate": round(hit_rate, 4),
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 4),
    }


def _empty_result() -> dict:
    return {
        "n_signals": 0,
        "n_filled_expected": 0.0,
        "total_gross_pnl": 0.0,
        "total_exit_fees": 0.0,
        "total_adverse_selection": 0.0,
        "total_net_pnl": 0.0,
        "total_expected_pnl": 0.0,
        "mean_expected_pnl": 0.0,
        "hit_rate": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Volume baseline backtest (pm-kcw.1)"
    )
    parser.add_argument(
        "--events-dir", type=Path, default=Path("data/events"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/backtest/volume_baseline_results.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-signals", type=int, default=485,
        help="Target number of signals per (vol_window, dt_seconds) combo",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Step 1: Find held-out events (same as TPP)
    # -----------------------------------------------------------------------
    print("Finding held-out events...")
    holdout_eids = []
    for p in sorted(args.events_dir.iterdir()):
        meta_path = p / "_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
        if end_s >= CUTOFF_EPOCH:
            holdout_eids.append(p.name)
    print(f"  Found {len(holdout_eids)} held-out events")

    # -----------------------------------------------------------------------
    # Step 2: Generate volume signals for all events
    # -----------------------------------------------------------------------
    print(f"\nGenerating volume signals...")
    all_signals = []
    t0 = time.time()

    for i, eid in enumerate(holdout_eids):
        sigs = generate_volume_signals_for_event(eid, args.events_dir)
        all_signals.extend(sigs)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(holdout_eids)}] {len(all_signals):,} candidates ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Total: {len(all_signals):,} candidates from {len(holdout_eids)} events ({elapsed:.1f}s)")

    if not all_signals:
        print("ERROR: No signals generated. Check event data.")
        return

    df_all = pl.DataFrame(all_signals)

    # -----------------------------------------------------------------------
    # Step 3: Apply threshold to select top signals by volume_strength
    # -----------------------------------------------------------------------
    # For each (volume_window, dt_seconds) combo, select top signals by
    # volume_strength to match ~485 signals (like TPP's threshold_pct=10).
    print(f"\nApplying thresholds (target: ~{args.target_signals} signals per combo)...")

    selected_signals = []
    for vol_win in VOLUME_WINDOWS:
        for dt_s in DT_VALUES:
            subset = df_all.filter(
                (pl.col("volume_window_s") == vol_win)
                & (pl.col("dt_seconds") == dt_s)
            )
            if subset.height == 0:
                continue

            # Sort by volume_strength descending, take top N
            n_take = min(args.target_signals, subset.height)
            top = subset.sort("volume_strength", descending=True).head(n_take)

            # Record the percentile cutoff
            cutoff = float(top["volume_strength"].min())

            selected_signals.append(top.with_columns(
                pl.lit(cutoff).alias("volume_threshold"),
            ))
            print(f"  vol_win={vol_win:>3}s dt={dt_s:>3}s: "
                  f"{top.height} signals (cutoff={cutoff:.3f})")

    if not selected_signals:
        print("ERROR: No signals after threshold. Check data.")
        return

    df_signals = pl.concat(selected_signals)
    print(f"  Total selected: {df_signals.height:,} signals")

    # -----------------------------------------------------------------------
    # Step 4: Run backtest with same pipeline as TPP
    # -----------------------------------------------------------------------
    print(f"\nRunning backtest...")

    # Compute directions from pred_bucket_pos (same as TPP)
    volume_dirs = np.where(
        df_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0
    )

    # Random baseline: coin-flip direction
    random_dirs = rng.choice([-1.0, 1.0], size=len(df_signals))

    signal_types = {
        "volume": volume_dirs,
        "random": random_dirs,
    }

    combos = list(itertools.product(VOLUME_WINDOWS, DT_VALUES, FILL_REGIMES.keys()))
    n_combos = len(combos) * len(signal_types)
    print(f"  {len(combos)} param combos × {len(signal_types)} signal types = {n_combos} total")

    results = []
    t0 = time.time()

    for sig_type, dirs_all in signal_types.items():
        for vol_win, dt_s, regime_name in combos:
            regime = FILL_REGIMES[regime_name]
            assumptions = FillAssumptions(
                adverse_selection_ticks=regime["adverse_ticks"],
                queue_position_frac=regime["queue_frac"],
            )

            # Filter signals for this (volume_window, dt_seconds)
            mask = (
                (df_signals["volume_window_s"] == vol_win)
                & (df_signals["dt_seconds"] == dt_s)
            )
            subset = df_signals.filter(mask)
            subset_idx = mask.to_numpy()
            subset_dirs = dirs_all[subset_idx]

            metrics = backtest_group(
                subset,
                subset_dirs,
                assumptions,
                taker_vol=regime["taker_volume"],
                resting_depth=regime["resting_depth"],
            )

            results.append({
                "signal_type": sig_type,
                "volume_window_s": vol_win,
                "dt_seconds": dt_s,
                "fill_regime": regime_name,
                "contracts": CONTRACTS,
                **metrics,
            })

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} results in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Step 5: Save results
    # -----------------------------------------------------------------------
    out_df = pl.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(args.out)
    print(f"\nSaved: {args.out} ({len(out_df)} rows)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VOLUME BASELINE BACKTEST SUMMARY")
    print("=" * 80)

    for sig_type in ["volume", "random"]:
        sub = out_df.filter(pl.col("signal_type") == sig_type)
        avg_exp_pnl = sub["total_expected_pnl"].mean()
        avg_hit = sub["hit_rate"].mean()
        avg_sharpe = sub["sharpe"].mean()
        avg_n = sub["n_signals"].mean()
        print(f"\n  {sig_type:>10s}: avg_n={avg_n:.0f}  avg_expected_pnl={avg_exp_pnl:>10.2f}  "
              f"avg_hit_rate={avg_hit:.3f}  avg_sharpe={avg_sharpe:.4f}")

    # Volume signal vs random by fill regime
    print("\n" + "-" * 80)
    print("VOLUME vs RANDOM (by fill regime)")
    print("-" * 80)
    for regime in FILL_REGIMES:
        print(f"\n  Fill regime: {regime}")
        for sig_type in ["volume", "random"]:
            sub = out_df.filter(
                (pl.col("signal_type") == sig_type)
                & (pl.col("fill_regime") == regime)
            )
            total_exp = sub["total_expected_pnl"].sum()
            avg_hit = sub["hit_rate"].mean()
            avg_sharpe = sub["sharpe"].mean()
            print(f"    {sig_type:>10s}: total_exp_pnl={total_exp:>10.2f}  "
                  f"hit={avg_hit:.3f}  sharpe={avg_sharpe:.4f}")

    # Best combos for volume signal
    print("\n" + "-" * 80)
    print("TOP 5 VOLUME COMBOS (by expected P&L)")
    print("-" * 80)
    vol_rows = out_df.filter(pl.col("signal_type") == "volume").sort(
        "total_expected_pnl", descending=True
    )
    for row in vol_rows.head(5).iter_rows(named=True):
        print(f"  vol_win={row['volume_window_s']:>3}s dt={row['dt_seconds']:>3}s "
              f"fill={row['fill_regime']:<13s} "
              f"exp_pnl={row['total_expected_pnl']:>8.2f}  "
              f"hit={row['hit_rate']:.3f}  sharpe={row['sharpe']:.4f}  "
              f"n={row['n_signals']}")

    print("\nBOTTOM 5 VOLUME COMBOS (by expected P&L)")
    print("-" * 80)
    for row in vol_rows.tail(5).iter_rows(named=True):
        print(f"  vol_win={row['volume_window_s']:>3}s dt={row['dt_seconds']:>3}s "
              f"fill={row['fill_regime']:<13s} "
              f"exp_pnl={row['total_expected_pnl']:>8.2f}  "
              f"hit={row['hit_rate']:.3f}  sharpe={row['sharpe']:.4f}  "
              f"n={row['n_signals']}")


if __name__ == "__main__":
    main()
