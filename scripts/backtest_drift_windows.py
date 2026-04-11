"""Drift window analysis at 2/5/15/30 min post-fill (pm-kcw.3).

Re-runs adverse selection analysis at longer horizons for both TPP and
volume baseline signals. Conditional on direction (BUY/SELL).

Uses EXISTING signal data — goes back to raw event parquets to compute
price changes at extended look-ahead windows.

Drift windows: 120s (2min), 300s (5min), 900s (15min), 1800s (30min)
Signal types: TPP model, volume baseline, random baseline
Direction split: BUY (+1) vs SELL (-1) vs ALL

Output: data/backtest/drift_window_results.parquet
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.costs.fees import taker_fee
from src.costs.fills import FillAssumptions, fill_probability, adverse_selection_cost

# ---------------------------------------------------------------------------
# Config
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
THETA = 0.050

# Extended drift windows (seconds).
DRIFT_WINDOWS = [120, 300, 900, 1800]

# Volume lookback windows (from volume baseline).
VOLUME_WINDOWS = [30, 60, 300]


# ---------------------------------------------------------------------------
# Extend signals with longer look-ahead from raw event data
# ---------------------------------------------------------------------------

def extend_price_changes(
    signals_df: pl.DataFrame,
    events_dir: Path,
    drift_windows: list[int],
) -> pl.DataFrame:
    """For each signal, go back to raw event data and compute price change
    at each drift window. Returns a new DataFrame with one row per
    (signal, drift_window) combination.

    signals_df must have: event_id, timestamp_ms, current_price, pred_bucket_pos
    """
    event_ids = signals_df["event_id"].unique().to_list()
    print(f"  Loading event data for {len(event_ids)} events...")

    # Cache event data: event_id -> (timestamps_ms, prices)
    event_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for eid in event_ids:
        parquet_path = events_dir / str(eid) / "events.parquet"
        if not parquet_path.exists():
            continue
        edf = pl.read_parquet(parquet_path).sort("seq")
        event_cache[eid] = (
            edf["timestamp_ms"].to_numpy().astype(np.float64),
            edf["price"].to_numpy().astype(np.float64),
        )

    print(f"  Loaded {len(event_cache)} events into cache")

    # For each signal, compute price_change at each drift window
    rows = []
    sig_ts = signals_df["timestamp_ms"].to_numpy()
    sig_eids = signals_df["event_id"].to_list()
    sig_prices = signals_df["current_price"].to_numpy()
    sig_preds = signals_df["pred_bucket_pos"].to_numpy()

    # Carry over extra columns if they exist
    has_city = "city" in signals_df.columns
    has_vol_win = "volume_window_s" in signals_df.columns
    has_threshold = "threshold_pct" in signals_df.columns
    has_vol_strength = "volume_strength" in signals_df.columns

    sig_cities = signals_df["city"].to_list() if has_city else [None] * len(sig_ts)
    sig_vol_wins = (
        signals_df["volume_window_s"].to_numpy() if has_vol_win else None
    )
    sig_thresholds = (
        signals_df["threshold_pct"].to_numpy() if has_threshold else None
    )

    n_signals = len(sig_ts)
    n_missing = 0

    for i in range(n_signals):
        eid = sig_eids[i]
        if eid not in event_cache:
            n_missing += 1
            continue

        ts_arr, price_arr = event_cache[eid]
        ts_now = float(sig_ts[i])
        cur_price = float(sig_prices[i])

        for dt_s in drift_windows:
            dt_ms = dt_s * 1000
            # Find trades within [ts_now, ts_now + dt_ms]
            future_mask = (ts_arr > ts_now) & (ts_arr <= ts_now + dt_ms)
            n_future = int(future_mask.sum())

            if n_future > 0:
                future_prices = price_arr[future_mask]
                price_at_end = float(future_prices[-1])
                price_change = price_at_end - cur_price
                max_price = float(future_prices.max())
                min_price = float(future_prices.min())
            else:
                price_change = 0.0
                max_price = cur_price
                min_price = cur_price

            row = {
                "event_id": eid,
                "city": sig_cities[i],
                "timestamp_ms": int(sig_ts[i]),
                "dt_seconds": dt_s,
                "pred_bucket_pos": float(sig_preds[i]),
                "current_price": cur_price,
                "price_change": round(price_change, 6),
                "max_price_in_window": round(max_price, 4),
                "min_price_in_window": round(min_price, 4),
                "n_future_events": n_future,
            }
            if has_vol_win and sig_vol_wins is not None:
                row["volume_window_s"] = int(sig_vol_wins[i])
            if has_threshold and sig_thresholds is not None:
                row["threshold_pct"] = int(sig_thresholds[i])

            rows.append(row)

    if n_missing > 0:
        print(f"  Warning: {n_missing} signals had missing event data")

    print(f"  Extended to {len(rows)} (signal, dt) combinations")
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_group(
    df: pl.DataFrame,
    directions: np.ndarray,
    fill_assumptions: FillAssumptions,
    taker_vol: float,
    resting_depth: float,
    contracts: float = CONTRACTS,
    theta: float = THETA,
) -> dict:
    """Run backtest on a group of signals. Returns aggregate metrics."""
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

    # Adverse selection diagnostic: mean absolute price drift
    mean_abs_drift = float(np.abs(price_changes).mean())
    # Directional drift: mean signed drift (positive = price went up)
    mean_signed_drift = float(price_changes.mean())
    # Drift against us: mean of (direction * -price_change) — positive = moved against
    mean_drift_against = float((-directions * price_changes).mean()) if n > 0 else 0.0

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
        "mean_abs_drift": round(mean_abs_drift, 6),
        "mean_signed_drift": round(mean_signed_drift, 6),
        "mean_drift_against": round(mean_drift_against, 6),
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
        "mean_abs_drift": 0.0,
        "mean_signed_drift": 0.0,
        "mean_drift_against": 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Drift window analysis at 2/5/15/30 min (pm-kcw.3)"
    )
    parser.add_argument(
        "--tpp-signals", type=Path, default=Path("data/signals/signals.parquet"),
    )
    parser.add_argument(
        "--events-dir", type=Path, default=Path("data/events"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/backtest/drift_window_results.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-volume-signals", type=int, default=485,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ===================================================================
    # PART A: TPP signals at longer drift windows
    # ===================================================================
    print("=" * 80)
    print("PART A: TPP SIGNALS — EXTENDED DRIFT WINDOWS")
    print("=" * 80)

    print("\nLoading TPP signals...")
    tpp_raw = pl.read_parquet(args.tpp_signals)
    # De-duplicate: use threshold_pct=10 and dt=60s as reference signals
    # (we'll recompute price changes at all drift windows)
    tpp_base = tpp_raw.filter(
        (pl.col("threshold_pct") == 10) & (pl.col("dt_seconds") == 60)
    ).unique(subset=["event_id", "timestamp_ms"])
    print(f"  {tpp_base.height} unique TPP signal points (thr=10, dt=60)")

    print("\nExtending TPP signals to drift windows {DRIFT_WINDOWS}...")
    t0 = time.time()
    tpp_extended = extend_price_changes(tpp_base, args.events_dir, DRIFT_WINDOWS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ===================================================================
    # PART B: Volume baseline signals at longer drift windows
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART B: VOLUME BASELINE SIGNALS — EXTENDED DRIFT WINDOWS")
    print("=" * 80)

    print("\nGenerating volume signals with extended drift windows...")
    from scripts.backtest_volume_baseline import (
        generate_volume_signals_for_event,
        CUTOFF_EPOCH,
    )

    # Find held-out events
    holdout_eids = []
    for p in sorted(args.events_dir.iterdir()):
        meta_path = p / "_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
        if end_s >= CUTOFF_EPOCH:
            holdout_eids.append(p.name)
    print(f"  {len(holdout_eids)} held-out events")

    t0 = time.time()
    vol_signals = []
    for i, eid in enumerate(holdout_eids):
        sigs = generate_volume_signals_for_event(
            eid, args.events_dir,
            volume_windows=VOLUME_WINDOWS,
            dt_values=DRIFT_WINDOWS,  # Use extended windows
        )
        vol_signals.extend(sigs)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(holdout_eids)}] {len(vol_signals):,} candidates")

    elapsed = time.time() - t0
    print(f"  {len(vol_signals):,} candidates in {elapsed:.1f}s")

    if not vol_signals:
        print("ERROR: No volume signals generated")
        return

    vol_all = pl.DataFrame(vol_signals)

    # Apply threshold (top N by volume_strength per combo)
    vol_selected = []
    for vol_win in VOLUME_WINDOWS:
        for dt_s in DRIFT_WINDOWS:
            subset = vol_all.filter(
                (pl.col("volume_window_s") == vol_win)
                & (pl.col("dt_seconds") == dt_s)
            )
            if subset.height == 0:
                continue
            n_take = min(args.target_volume_signals, subset.height)
            top = subset.sort("volume_strength", descending=True).head(n_take)
            vol_selected.append(top)

    vol_df = pl.concat(vol_selected) if vol_selected else pl.DataFrame()
    print(f"  Selected: {vol_df.height} volume signals after threshold")

    # ===================================================================
    # PART C: Run backtest — direction-conditional
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART C: DIRECTION-CONDITIONAL BACKTEST")
    print("=" * 80)

    results = []

    # --- TPP signals ---
    print("\nBacktesting TPP signals...")
    tpp_model_dirs = np.where(
        tpp_extended["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0
    )
    tpp_random_dirs = rng.choice([-1.0, 1.0], size=len(tpp_extended))

    tpp_signal_types = {
        "tpp_model": tpp_model_dirs,
        "tpp_random": tpp_random_dirs,
    }

    for sig_type, dirs_all in tpp_signal_types.items():
        for dt_s in DRIFT_WINDOWS:
            for regime_name, regime in FILL_REGIMES.items():
                assumptions = FillAssumptions(
                    adverse_selection_ticks=regime["adverse_ticks"],
                    queue_position_frac=regime["queue_frac"],
                )

                mask = tpp_extended["dt_seconds"] == dt_s
                subset = tpp_extended.filter(mask)
                subset_dirs = dirs_all[mask.to_numpy()]

                # ALL directions
                metrics = backtest_group(
                    subset, subset_dirs, assumptions,
                    taker_vol=regime["taker_volume"],
                    resting_depth=regime["resting_depth"],
                )
                results.append({
                    "source": "tpp",
                    "signal_type": sig_type,
                    "direction_filter": "all",
                    "dt_seconds": dt_s,
                    "volume_window_s": 0,
                    "fill_regime": regime_name,
                    "contracts": CONTRACTS,
                    **metrics,
                })

                # BUY only (direction = +1)
                buy_mask = subset_dirs > 0
                if buy_mask.sum() > 0:
                    buy_metrics = backtest_group(
                        subset.filter(pl.Series(buy_mask)),
                        subset_dirs[buy_mask],
                        assumptions,
                        taker_vol=regime["taker_volume"],
                        resting_depth=regime["resting_depth"],
                    )
                    results.append({
                        "source": "tpp",
                        "signal_type": sig_type,
                        "direction_filter": "buy",
                        "dt_seconds": dt_s,
                        "volume_window_s": 0,
                        "fill_regime": regime_name,
                        "contracts": CONTRACTS,
                        **buy_metrics,
                    })

                # SELL only (direction = -1)
                sell_mask = subset_dirs < 0
                if sell_mask.sum() > 0:
                    sell_metrics = backtest_group(
                        subset.filter(pl.Series(sell_mask)),
                        subset_dirs[sell_mask],
                        assumptions,
                        taker_vol=regime["taker_volume"],
                        resting_depth=regime["resting_depth"],
                    )
                    results.append({
                        "source": "tpp",
                        "signal_type": sig_type,
                        "direction_filter": "sell",
                        "dt_seconds": dt_s,
                        "volume_window_s": 0,
                        "fill_regime": regime_name,
                        "contracts": CONTRACTS,
                        **sell_metrics,
                    })

    # --- Volume signals ---
    if vol_df.height > 0:
        print("\nBacktesting volume signals...")
        vol_model_dirs = np.where(
            vol_df["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0
        )
        vol_random_dirs = rng.choice([-1.0, 1.0], size=len(vol_df))

        vol_signal_types = {
            "volume_model": vol_model_dirs,
            "volume_random": vol_random_dirs,
        }

        for sig_type, dirs_all in vol_signal_types.items():
            for vol_win in VOLUME_WINDOWS:
                for dt_s in DRIFT_WINDOWS:
                    for regime_name, regime in FILL_REGIMES.items():
                        assumptions = FillAssumptions(
                            adverse_selection_ticks=regime["adverse_ticks"],
                            queue_position_frac=regime["queue_frac"],
                        )

                        mask = (
                            (vol_df["volume_window_s"] == vol_win)
                            & (vol_df["dt_seconds"] == dt_s)
                        )
                        subset = vol_df.filter(mask)
                        subset_idx = mask.to_numpy()
                        subset_dirs = dirs_all[subset_idx]

                        if subset.height == 0:
                            continue

                        # ALL
                        metrics = backtest_group(
                            subset, subset_dirs, assumptions,
                            taker_vol=regime["taker_volume"],
                            resting_depth=regime["resting_depth"],
                        )
                        results.append({
                            "source": "volume",
                            "signal_type": sig_type,
                            "direction_filter": "all",
                            "dt_seconds": dt_s,
                            "volume_window_s": vol_win,
                            "fill_regime": regime_name,
                            "contracts": CONTRACTS,
                            **metrics,
                        })

                        # BUY
                        buy_mask = subset_dirs > 0
                        if buy_mask.sum() > 0:
                            buy_metrics = backtest_group(
                                subset.filter(pl.Series(buy_mask)),
                                subset_dirs[buy_mask],
                                assumptions,
                                taker_vol=regime["taker_volume"],
                                resting_depth=regime["resting_depth"],
                            )
                            results.append({
                                "source": "volume",
                                "signal_type": sig_type,
                                "direction_filter": "buy",
                                "dt_seconds": dt_s,
                                "volume_window_s": vol_win,
                                "fill_regime": regime_name,
                                "contracts": CONTRACTS,
                                **buy_metrics,
                            })

                        # SELL
                        sell_mask = subset_dirs < 0
                        if sell_mask.sum() > 0:
                            sell_metrics = backtest_group(
                                subset.filter(pl.Series(sell_mask)),
                                subset_dirs[sell_mask],
                                assumptions,
                                taker_vol=regime["taker_volume"],
                                resting_depth=regime["resting_depth"],
                            )
                            results.append({
                                "source": "volume",
                                "signal_type": sig_type,
                                "direction_filter": "sell",
                                "dt_seconds": dt_s,
                                "volume_window_s": vol_win,
                                "fill_regime": regime_name,
                                "contracts": CONTRACTS,
                                **sell_metrics,
                            })

    # ===================================================================
    # Save and summarize
    # ===================================================================
    print(f"\n  Total results: {len(results)}")

    out_df = pl.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(args.out)
    print(f"  Saved: {args.out} ({len(out_df)} rows)")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("DRIFT WINDOW ANALYSIS SUMMARY")
    print("=" * 80)

    # --- TPP summary ---
    print("\n--- TPP MODEL: Adverse Selection by Drift Window ---")
    print(f"{'dt':>6s}  {'dir':>5s}  {'regime':<13s}  {'n':>5s}  "
          f"{'hit_rate':>8s}  {'sharpe':>8s}  {'exp_pnl':>10s}  "
          f"{'abs_drift':>10s}  {'drift_against':>14s}")
    print("-" * 95)

    tpp_rows = out_df.filter(
        (pl.col("source") == "tpp") & (pl.col("signal_type") == "tpp_model")
    ).sort(["dt_seconds", "direction_filter", "fill_regime"])

    for row in tpp_rows.iter_rows(named=True):
        print(f"{row['dt_seconds']:>5}s  {row['direction_filter']:>5s}  "
              f"{row['fill_regime']:<13s}  {row['n_signals']:>5}  "
              f"{row['hit_rate']:>8.4f}  {row['sharpe']:>8.4f}  "
              f"{row['total_expected_pnl']:>10.2f}  "
              f"{row['mean_abs_drift']:>10.6f}  "
              f"{row['mean_drift_against']:>14.6f}")

    # --- Volume summary (moderate regime only for brevity) ---
    print("\n--- VOLUME MODEL: Adverse Selection by Drift Window (moderate regime) ---")
    print(f"{'vol_win':>7s}  {'dt':>6s}  {'dir':>5s}  {'n':>5s}  "
          f"{'hit_rate':>8s}  {'sharpe':>8s}  {'exp_pnl':>10s}  "
          f"{'abs_drift':>10s}  {'drift_against':>14s}")
    print("-" * 90)

    vol_rows = out_df.filter(
        (pl.col("source") == "volume")
        & (pl.col("signal_type") == "volume_model")
        & (pl.col("fill_regime") == "moderate")
    ).sort(["volume_window_s", "dt_seconds", "direction_filter"])

    for row in vol_rows.iter_rows(named=True):
        print(f"{row['volume_window_s']:>6}s  {row['dt_seconds']:>5}s  "
              f"{row['direction_filter']:>5s}  {row['n_signals']:>5}  "
              f"{row['hit_rate']:>8.4f}  {row['sharpe']:>8.4f}  "
              f"{row['total_expected_pnl']:>10.2f}  "
              f"{row['mean_abs_drift']:>10.6f}  "
              f"{row['mean_drift_against']:>14.6f}")

    # --- Direction asymmetry summary ---
    print("\n--- DIRECTION ASYMMETRY (moderate regime, all signal types) ---")
    print(f"{'source':>8s}  {'sig_type':>14s}  {'dt':>6s}  "
          f"{'buy_hit':>8s}  {'sell_hit':>9s}  "
          f"{'buy_sharpe':>10s}  {'sell_sharpe':>11s}  "
          f"{'buy_drift_ag':>12s}  {'sell_drift_ag':>13s}")
    print("-" * 110)

    for source in ["tpp", "volume"]:
        sig_types = (
            ["tpp_model"] if source == "tpp"
            else ["volume_model"]
        )
        for sig_type in sig_types:
            for dt_s in DRIFT_WINDOWS:
                buy_row = out_df.filter(
                    (pl.col("source") == source)
                    & (pl.col("signal_type") == sig_type)
                    & (pl.col("fill_regime") == "moderate")
                    & (pl.col("dt_seconds") == dt_s)
                    & (pl.col("direction_filter") == "buy")
                )
                sell_row = out_df.filter(
                    (pl.col("source") == source)
                    & (pl.col("signal_type") == sig_type)
                    & (pl.col("fill_regime") == "moderate")
                    & (pl.col("dt_seconds") == dt_s)
                    & (pl.col("direction_filter") == "sell")
                )
                if buy_row.height == 0 or sell_row.height == 0:
                    continue
                b = buy_row.row(0, named=True)
                s = sell_row.row(0, named=True)
                print(f"{source:>8s}  {sig_type:>14s}  {dt_s:>5}s  "
                      f"{b['hit_rate']:>8.4f}  {s['hit_rate']:>9.4f}  "
                      f"{b['sharpe']:>10.4f}  {s['sharpe']:>11.4f}  "
                      f"{b['mean_drift_against']:>12.6f}  "
                      f"{s['mean_drift_against']:>13.6f}")

    # Best combos
    print("\n--- TOP 10 COMBOS BY SHARPE (all sources, moderate regime) ---")
    top = out_df.filter(
        (pl.col("fill_regime") == "moderate")
        & (pl.col("direction_filter") == "all")
        & (~pl.col("signal_type").str.contains("random"))
    ).sort("sharpe", descending=True).head(10)

    for row in top.iter_rows(named=True):
        vw = f"vw={row['volume_window_s']}s" if row['volume_window_s'] > 0 else "      "
        print(f"  {row['source']:>7s}  {row['signal_type']:>14s}  "
              f"dt={row['dt_seconds']:>5}s  {vw}  "
              f"sharpe={row['sharpe']:>8.4f}  "
              f"hit={row['hit_rate']:.4f}  "
              f"exp_pnl={row['total_expected_pnl']:>10.2f}  "
              f"n={row['n_signals']}")


if __name__ == "__main__":
    main()
