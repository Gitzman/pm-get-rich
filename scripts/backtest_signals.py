"""Backtest TPP signals with cost model, parameter sweep, and baselines (pm-kxg.3).

For each signal from generate_signals_v2, simulates a maker entry / taker exit
round-trip trade using the cost model from src/costs/.

Sweeps parameters (~108 combos):
  - dt_seconds: [15, 30, 60, 120]
  - threshold_pct: [1, 5, 10]
  - fill_regime: [conservative, moderate, optimistic]
  - signal_type: [model, random, shuffled]

Baselines:
  - random: replace model direction with coin-flip direction
  - shuffled: shuffle pred_bucket_pos within each (event_id, dt_seconds) group,
    destroying predictive content while preserving distribution shape

Output: data/backtest_results.parquet
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.costs.fees import taker_fee
from src.costs.fills import FillAssumptions, fill_probability, adverse_selection_cost


# ---------------------------------------------------------------------------
# Fill regimes: represent different assumptions about market microstructure
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

# Order size for all combos (fixed; gross P&L and costs both scale linearly).
CONTRACTS = 100


def compute_signal_direction(pred_bucket_pos: np.ndarray) -> np.ndarray:
    """Model signal: BUY if pred_bucket_pos > 0.5, SELL otherwise.

    Returns +1 (long) or -1 (short).
    """
    return np.where(pred_bucket_pos > 0.5, 1.0, -1.0)


def make_random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random baseline: coin-flip direction."""
    return rng.choice([-1.0, 1.0], size=n)


def make_shuffled_directions(
    df: pl.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Bucket-shuffled baseline: shuffle pred_bucket_pos within each event,
    then derive direction. Preserves marginal distribution but destroys
    temporal signal.
    """
    bucket_pos = df["pred_bucket_pos"].to_numpy().copy()
    event_ids = df["event_id"].to_numpy()

    for eid in np.unique(event_ids):
        mask = event_ids == eid
        bucket_pos[mask] = rng.permutation(bucket_pos[mask])

    return compute_signal_direction(bucket_pos)


def backtest_group(
    df: pl.DataFrame,
    directions: np.ndarray,
    fill_assumptions: FillAssumptions,
    taker_vol: float,
    resting_depth: float,
    contracts: float = CONTRACTS,
    theta: float = 0.050,
) -> dict:
    """Run backtest on a group of signals with given directions and fill params.

    Returns aggregate metrics for the group.
    """
    prices = df["current_price"].to_numpy()
    price_changes = df["price_change"].to_numpy()
    n = len(prices)

    if n == 0:
        return _empty_result()

    # Per-signal calculations
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

        # Gross P&L: direction × price_change × contracts
        gross_pnl[i] = d * pc * contracts

        # Fill probability
        fill_probs[i] = fill_probability(
            taker_volume_at_price=taker_vol,
            resting_depth_at_price=resting_depth,
            our_size=contracts,
            assumptions=fill_assumptions,
        )

        # Exit fee: taker fee at exit price (early exit at p + price_change)
        exit_price = np.clip(p + pc, 0.01, 0.99)
        exit_fees[i] = taker_fee(float(exit_price), contracts, theta)

        # Adverse selection cost
        adv_sel_costs[i] = adverse_selection_cost(
            contracts, assumptions=fill_assumptions
        )

        # Net P&L = gross - costs
        net_pnl[i] = gross_pnl[i] - exit_fees[i] - adv_sel_costs[i]

        # Expected P&L = fill_prob × net_pnl
        expected_pnl[i] = fill_probs[i] * net_pnl[i]

    # Aggregate metrics
    n_filled = float(fill_probs.sum())
    total_gross = float(gross_pnl.sum())
    total_fees = float(exit_fees.sum())
    total_adv_sel = float(adv_sel_costs.sum())
    total_net = float(net_pnl.sum())
    total_expected = float(expected_pnl.sum())

    # Direction accuracy: did the direction match the sign of price_change?
    correct_dir = np.sum((directions * price_changes) > 0)
    flat = np.sum(np.abs(price_changes) < 1e-6)
    hit_rate = float(correct_dir) / max(n - int(flat), 1)

    # Sharpe on expected P&L per signal
    if n > 1 and expected_pnl.std() > 0:
        sharpe = float(expected_pnl.mean() / expected_pnl.std())
    else:
        sharpe = 0.0

    # Win rate: fraction of signals with positive expected P&L
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


def main():
    parser = argparse.ArgumentParser(
        description="Backtest signals with cost model + baselines (pm-kxg.3)"
    )
    parser.add_argument(
        "--signals", type=Path, default=Path("data/signals/signals.parquet")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/backtest_results.parquet")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load signals
    print("Loading signals...")
    df = pl.read_parquet(args.signals)
    print(f"  {len(df):,} signals, {df['event_id'].n_unique()} events")

    dt_values = sorted(df["dt_seconds"].unique().to_list())
    thr_values = sorted(df["threshold_pct"].unique().to_list())
    print(f"  dt_seconds: {dt_values}")
    print(f"  threshold_pct: {thr_values}")

    # Pre-compute directions for each signal type
    model_dirs = compute_signal_direction(df["pred_bucket_pos"].to_numpy())
    random_dirs = make_random_directions(len(df), rng)
    shuffled_dirs = make_shuffled_directions(df, rng)

    signal_types = {
        "model": model_dirs,
        "random": random_dirs,
        "shuffled": shuffled_dirs,
    }

    # Parameter grid
    combos = list(itertools.product(dt_values, thr_values, FILL_REGIMES.keys()))
    n_combos = len(combos) * len(signal_types)
    print(f"\nSweeping {len(combos)} param combos × {len(signal_types)} signal types = {n_combos} total")

    results = []
    t0 = time.time()

    for sig_type, dirs in signal_types.items():
        for dt_s, thr_pct, regime_name in combos:
            regime = FILL_REGIMES[regime_name]

            assumptions = FillAssumptions(
                adverse_selection_ticks=regime["adverse_ticks"],
                queue_position_frac=regime["queue_frac"],
            )

            # Filter signals for this (dt_seconds, threshold_pct)
            mask = (df["dt_seconds"] == dt_s) & (df["threshold_pct"] == thr_pct)
            subset = df.filter(mask)
            subset_idx = mask.to_numpy()
            subset_dirs = dirs[subset_idx]

            metrics = backtest_group(
                subset,
                subset_dirs,
                assumptions,
                taker_vol=regime["taker_volume"],
                resting_depth=regime["resting_depth"],
            )

            results.append({
                "signal_type": sig_type,
                "dt_seconds": dt_s,
                "threshold_pct": thr_pct,
                "fill_regime": regime_name,
                "contracts": CONTRACTS,
                **metrics,
            })

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} results in {elapsed:.1f}s")

    # Build output dataframe
    out_df = pl.DataFrame(results)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(args.out)
    print(f"Saved: {args.out} ({len(out_df)} rows)")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    for sig_type in ["model", "random", "shuffled"]:
        sub = out_df.filter(pl.col("signal_type") == sig_type)
        avg_exp_pnl = sub["total_expected_pnl"].mean()
        avg_hit = sub["hit_rate"].mean()
        avg_sharpe = sub["sharpe"].mean()
        print(f"\n  {sig_type:>10s}: avg_expected_pnl={avg_exp_pnl:>10.2f}  "
              f"avg_hit_rate={avg_hit:.3f}  avg_sharpe={avg_sharpe:.4f}")

    # Model vs baselines comparison
    print("\n" + "-" * 80)
    print("MODEL vs BASELINES (by fill regime)")
    print("-" * 80)
    for regime in FILL_REGIMES:
        print(f"\n  Fill regime: {regime}")
        for sig_type in ["model", "random", "shuffled"]:
            sub = out_df.filter(
                (pl.col("signal_type") == sig_type)
                & (pl.col("fill_regime") == regime)
            )
            total_exp = sub["total_expected_pnl"].sum()
            avg_hit = sub["hit_rate"].mean()
            avg_sharpe = sub["sharpe"].mean()
            print(f"    {sig_type:>10s}: total_exp_pnl={total_exp:>10.2f}  "
                  f"hit={avg_hit:.3f}  sharpe={avg_sharpe:.4f}")

    # Best and worst combos for model
    print("\n" + "-" * 80)
    print("TOP 5 MODEL COMBOS (by expected P&L)")
    print("-" * 80)
    model_rows = out_df.filter(pl.col("signal_type") == "model").sort(
        "total_expected_pnl", descending=True
    )
    for row in model_rows.head(5).iter_rows(named=True):
        print(f"  dt={row['dt_seconds']:>3}s thr={row['threshold_pct']:>2}% "
              f"fill={row['fill_regime']:<13s} "
              f"exp_pnl={row['total_expected_pnl']:>8.2f}  "
              f"hit={row['hit_rate']:.3f}  sharpe={row['sharpe']:.4f}  "
              f"n={row['n_signals']}")

    print("\nBOTTOM 5 MODEL COMBOS (by expected P&L)")
    print("-" * 80)
    for row in model_rows.tail(5).iter_rows(named=True):
        print(f"  dt={row['dt_seconds']:>3}s thr={row['threshold_pct']:>2}% "
              f"fill={row['fill_regime']:<13s} "
              f"exp_pnl={row['total_expected_pnl']:>8.2f}  "
              f"hit={row['hit_rate']:.3f}  sharpe={row['sharpe']:.4f}  "
              f"n={row['n_signals']}")


if __name__ == "__main__":
    main()
