"""TPP vs Volume baseline comparison (pm-kcw.2).

Side-by-side comparison:
  1. Mean P&L, bootstrap CI, fill rate, win rate
  2. Per-trade matched comparison where both strategies fired
  3. Stratification by market (city) + time-of-day
  4. Cumulative P&L curves on same chart

Uses:
  - TPP signals: data/signals/signals.parquet
  - Volume signals: regenerated from events data (same logic as backtest_volume_baseline.py)
  - Same cost model, fill regimes, and metrics as both backtests

Output:
  - data/reports/tpp_vs_volume/  (charts + summary JSON)
  - data/backtest/tpp_vs_volume_comparison.parquet  (per-trade results)
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.costs.fees import taker_fee
from src.costs.fills import FillAssumptions, fill_probability, adverse_selection_cost

# ---------------------------------------------------------------------------
# Config (shared with both backtests)
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
N_BOOTSTRAP = 5000
SEED = 42

CUTOFF_EPOCH = datetime.datetime(2026, 3, 25, tzinfo=datetime.timezone.utc).timestamp()

# Volume signal config (best from pm-kcw.1: 60s window)
VOLUME_WINDOW_S = 60
DT_VALUES = [15, 30, 60, 120]

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Per-trade P&L computation
# ---------------------------------------------------------------------------

def compute_per_trade_pnl(
    df: pl.DataFrame,
    directions: np.ndarray,
    regime_name: str,
    contracts: float = CONTRACTS,
    theta: float = THETA,
) -> np.ndarray:
    """Compute per-trade expected P&L array."""
    regime = FILL_REGIMES[regime_name]
    assumptions = FillAssumptions(
        adverse_selection_ticks=regime["adverse_ticks"],
        queue_position_frac=regime["queue_frac"],
    )
    taker_vol = regime["taker_volume"]
    resting_depth = regime["resting_depth"]

    prices = df["current_price"].to_numpy()
    price_changes = df["price_change"].to_numpy()
    n = len(prices)

    gross_pnl = directions * price_changes * contracts
    fill_prob = fill_probability(taker_vol, resting_depth, contracts, assumptions)
    exit_prices = np.clip(prices + price_changes, 0.01, 0.99)
    exit_fees = np.array([taker_fee(float(ep), contracts, theta) for ep in exit_prices])
    adv_sel = adverse_selection_cost(contracts, assumptions=assumptions)
    net_pnl = gross_pnl - exit_fees - adv_sel
    expected_pnl = fill_prob * net_pnl

    return expected_pnl


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    point = stat_fn(values)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


def sharpe_fn(x: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std())


# ---------------------------------------------------------------------------
# Volume signal generation (from backtest_volume_baseline.py)
# ---------------------------------------------------------------------------

def generate_volume_signals_for_event(
    event_id: str,
    events_dir: Path,
    volume_window_s: int = VOLUME_WINDOW_S,
    dt_values: list[int] = DT_VALUES,
    step_frac: int = 4,
) -> list[dict]:
    """Generate volume-spike signals for one event (single window)."""
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

    side_signs = np.array([1.0 if s == "BUY" else -1.0 for s in sides])
    signed_sizes = sizes * side_signs

    split = int(0.8 * N)
    if N - split < 20:
        return []

    vol_win_ms = volume_window_s * 1000

    # First pass: compute rolling volume at every step for percentile ranking.
    all_volumes = []
    step_size = max(1, (N - split) // (4 * step_frac))

    for i in range(split, N, step_size):
        ts_now = timestamps_ms[i]
        lookback_start = ts_now - vol_win_ms
        mask = timestamps_ms[:i] >= lookback_start
        vol = float(sizes[:i][mask].sum())
        all_volumes.append(vol)

    if not all_volumes:
        return []

    all_volumes_arr = np.array(all_volumes)

    signals = []
    for i in range(split, N, step_size):
        ts_now = timestamps_ms[i]
        lookback_start = ts_now - vol_win_ms

        mask = timestamps_ms[:i] >= lookback_start
        window_sizes = sizes[:i][mask]
        window_signed = signed_sizes[:i][mask]

        vol = float(window_sizes.sum())
        n_trades = int(mask.sum())

        if n_trades < 2 or vol < 1e-6:
            continue

        volume_strength = float((all_volumes_arr <= vol).sum()) / len(all_volumes_arr)
        net_signed = float(window_signed.sum())
        direction = 1.0 if net_signed > 0 else -1.0
        pred_bucket_pos = 0.5 + direction * volume_strength * 0.5

        current_price = float(prices[i - 1])

        for dt_s in dt_values:
            dt_ms = dt_s * 1000
            future_mask = (timestamps_ms[i:] - ts_now) <= dt_ms
            n_future = int(future_mask.sum())

            if n_future > 0:
                future_prices = prices[i:i + n_future]
                price_at_end = float(future_prices[-1])
                price_change = price_at_end - current_price
            else:
                price_change = 0.0

            signals.append({
                "event_id": event_id,
                "city": city,
                "timestamp_ms": int(ts_now),
                "dt_seconds": dt_s,
                "volume_strength": round(volume_strength, 4),
                "pred_bucket_pos": round(pred_bucket_pos, 6),
                "current_price": round(current_price, 4),
                "price_change": round(price_change, 6),
                "n_future_events": n_future,
            })

    return signals


# ---------------------------------------------------------------------------
# Section 1: Aggregate comparison — mean P&L, bootstrap CI, fill rate, win rate
# ---------------------------------------------------------------------------

def section_aggregate(
    tpp_signals: pl.DataFrame,
    vol_signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
) -> dict:
    print("\n" + "=" * 80)
    print("SECTION 1: AGGREGATE COMPARISON (TPP vs VOLUME)")
    print("=" * 80)

    regime = "moderate"

    results = {}
    for label, sigs in [("TPP", tpp_signals), ("Volume", vol_signals)]:
        dirs = np.where(sigs["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        epnl = compute_per_trade_pnl(sigs, dirs, regime)
        price_changes = sigs["price_change"].to_numpy()

        correct = (dirs * price_changes) > 0
        flat = np.abs(price_changes) < 1e-6
        non_flat = ~flat

        hit_vals = correct[non_flat].astype(float)
        win_vals = (epnl > 0).astype(float)

        metrics = {}
        for name, fn, vals in [
            ("mean_pnl", np.mean, epnl),
            ("total_pnl", np.sum, epnl),
            ("sharpe", sharpe_fn, epnl),
            ("hit_rate", np.mean, hit_vals),
            ("win_rate", np.mean, win_vals),
        ]:
            pt, lo, hi = bootstrap_ci(vals, fn, rng=rng)
            metrics[name] = {"point": pt, "ci_lo": lo, "ci_hi": hi}

        metrics["n_signals"] = len(sigs)
        metrics["fill_rate"] = float(
            fill_probability(
                FILL_REGIMES[regime]["taker_volume"],
                FILL_REGIMES[regime]["resting_depth"],
                CONTRACTS,
                FillAssumptions(
                    adverse_selection_ticks=FILL_REGIMES[regime]["adverse_ticks"],
                    queue_position_frac=FILL_REGIMES[regime]["queue_frac"],
                ),
            )
        )
        results[label] = metrics

        print(f"\n  {label} (n={len(sigs)}):")
        for name in ["mean_pnl", "total_pnl", "sharpe", "hit_rate", "win_rate"]:
            m = metrics[name]
            print(f"    {name:>15s}: {m['point']:>10.4f}  [{m['ci_lo']:>10.4f}, {m['ci_hi']:>10.4f}]")
        print(f"    {'fill_rate':>15s}: {metrics['fill_rate']:.4f}")

    # CI overlap check
    print("\n  --- CI Overlap ---")
    for metric_name in ["mean_pnl", "sharpe", "hit_rate"]:
        t = results["TPP"][metric_name]
        v = results["Volume"][metric_name]
        sep = t["ci_lo"] > v["ci_hi"] or v["ci_lo"] > t["ci_hi"]
        print(f"  {metric_name}: CIs {'SEPARATED' if sep else 'OVERLAP'}")

    # Figure: side-by-side CI bars
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric_key, metric_label in [
        (axes[0], "mean_pnl", "Mean Expected P&L ($)"),
        (axes[1], "sharpe", "Sharpe Ratio"),
        (axes[2], "hit_rate", "Hit Rate"),
        (axes[3], "win_rate", "Win Rate"),
    ]:
        labels_list = []
        points = []
        lows = []
        highs = []
        for sig_type in ["TPP", "Volume"]:
            ci = results[sig_type][metric_key]
            labels_list.append(f"{sig_type}\n(n={results[sig_type]['n_signals']})")
            points.append(ci["point"])
            lows.append(ci["point"] - ci["ci_lo"])
            highs.append(ci["ci_hi"] - ci["point"])

        colors = ["steelblue", "coral"]
        y_pos = np.arange(len(labels_list))
        ax.barh(y_pos, points, xerr=[lows, highs], color=colors, alpha=0.7,
                edgecolor="white", capsize=5, height=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel(metric_label)
        ax.set_title(metric_label)

    fig.suptitle("TPP vs Volume Baseline (Moderate Fill Regime, 95% Bootstrap CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "01_aggregate_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '01_aggregate_comparison.png'}")

    # By fill regime
    print("\n  --- All Fill Regimes ---")
    regime_results = []
    for reg_name in FILL_REGIMES:
        for label, sigs in [("TPP", tpp_signals), ("Volume", vol_signals)]:
            dirs = np.where(sigs["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
            epnl = compute_per_trade_pnl(sigs, dirs, reg_name)
            mean_ci = bootstrap_ci(epnl, np.mean, rng=rng)
            regime_results.append({
                "regime": reg_name, "strategy": label,
                "mean_pnl": mean_ci[0], "ci_lo": mean_ci[1], "ci_hi": mean_ci[2],
            })
            print(f"    {reg_name:<14s} {label:<8s}: "
                  f"mean={mean_ci[0]:>8.4f}  [{mean_ci[1]:>8.4f}, {mean_ci[2]:>8.4f}]")

    return results


# ---------------------------------------------------------------------------
# Section 2: Per-trade matched comparison
# ---------------------------------------------------------------------------

def section_matched(
    tpp_signals: pl.DataFrame,
    vol_signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
) -> dict:
    print("\n" + "=" * 80)
    print("SECTION 2: MATCHED TRADE COMPARISON")
    print("=" * 80)

    regime = "moderate"

    # Match on (event_id, timestamp_ms, dt_seconds)
    tpp_keys = tpp_signals.select("event_id", "timestamp_ms", "dt_seconds")
    vol_keys = vol_signals.select("event_id", "timestamp_ms", "dt_seconds")

    # Inner join to find exact matches
    matched = tpp_keys.join(vol_keys, on=["event_id", "timestamp_ms", "dt_seconds"], how="inner")
    n_matched = matched.height
    print(f"  Exact matches (event_id, timestamp_ms, dt_seconds): {n_matched}")

    if n_matched < 10:
        # Try approximate matching: same event + dt, closest timestamp within 5s
        print("  Too few exact matches. Trying approximate matching (within 5s)...")
        matched_rows = []
        for dt_s in DT_VALUES:
            tpp_dt = tpp_signals.filter(pl.col("dt_seconds") == dt_s)
            vol_dt = vol_signals.filter(pl.col("dt_seconds") == dt_s)

            for eid in tpp_dt["event_id"].unique().to_list():
                tpp_ev = tpp_dt.filter(pl.col("event_id") == eid).sort("timestamp_ms")
                vol_ev = vol_dt.filter(pl.col("event_id") == eid).sort("timestamp_ms")

                if vol_ev.height == 0:
                    continue

                tpp_ts = tpp_ev["timestamp_ms"].to_numpy()
                vol_ts = vol_ev["timestamp_ms"].to_numpy()

                for t_idx in range(len(tpp_ts)):
                    diffs = np.abs(vol_ts - tpp_ts[t_idx])
                    closest = int(np.argmin(diffs))
                    if diffs[closest] <= 5000:  # within 5 seconds
                        matched_rows.append({
                            "event_id": eid,
                            "dt_seconds": dt_s,
                            "tpp_idx": t_idx,
                            "vol_idx": closest,
                            "tpp_ts": int(tpp_ts[t_idx]),
                            "vol_ts": int(vol_ts[closest]),
                        })

        if matched_rows:
            matched_df = pl.DataFrame(matched_rows)
            n_matched = matched_df.height
            print(f"  Approximate matches (within 5s): {n_matched}")
        else:
            n_matched = 0

    if n_matched == 0:
        print("  No matched trades found. Comparing all signals independently.")
        print("  (TPP and Volume fire at different times/events — expected for different signals.)")

        # Still do a paired comparison on the same events
        shared_events = set(tpp_signals["event_id"].unique().to_list()) & \
                       set(vol_signals["event_id"].unique().to_list())
        print(f"  Shared events: {len(shared_events)}")

        # Per-event aggregate comparison
        tpp_event_pnl = {}
        vol_event_pnl = {}

        tpp_dirs = np.where(tpp_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        tpp_epnl = compute_per_trade_pnl(tpp_signals, tpp_dirs, regime)
        vol_dirs = np.where(vol_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        vol_epnl = compute_per_trade_pnl(vol_signals, vol_dirs, regime)

        tpp_eids = tpp_signals["event_id"].to_list()
        vol_eids = vol_signals["event_id"].to_list()

        for i, eid in enumerate(tpp_eids):
            tpp_event_pnl.setdefault(eid, []).append(tpp_epnl[i])
        for i, eid in enumerate(vol_eids):
            vol_event_pnl.setdefault(eid, []).append(vol_epnl[i])

        paired_tpp = []
        paired_vol = []
        event_labels = []
        for eid in sorted(shared_events):
            if eid in tpp_event_pnl and eid in vol_event_pnl:
                paired_tpp.append(np.mean(tpp_event_pnl[eid]))
                paired_vol.append(np.mean(vol_event_pnl[eid]))
                event_labels.append(eid)

        paired_tpp_arr = np.array(paired_tpp)
        paired_vol_arr = np.array(paired_vol)
        diff = paired_vol_arr - paired_tpp_arr

        print(f"\n  Per-event paired comparison ({len(paired_tpp)} events):")
        diff_ci = bootstrap_ci(diff, np.mean, rng=rng)
        print(f"    Mean diff (Vol - TPP): {diff_ci[0]:>8.4f}  [{diff_ci[1]:>8.4f}, {diff_ci[2]:>8.4f}]")
        vol_wins = (diff > 0).sum()
        print(f"    Volume wins: {vol_wins}/{len(diff)} ({vol_wins/len(diff):.1%})")

        # Scatter: per-event mean P&L
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(paired_tpp_arr, paired_vol_arr, alpha=0.4, s=20, color="steelblue")
        lim = max(abs(paired_tpp_arr).max(), abs(paired_vol_arr).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="TPP = Volume")
        ax.set_xlabel("TPP mean expected P&L per event ($)")
        ax.set_ylabel("Volume mean expected P&L per event ($)")
        ax.set_title(f"Per-Event Paired Comparison (n={len(paired_tpp)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "02_matched_scatter.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / '02_matched_scatter.png'}")

        return {"n_matched": 0, "n_shared_events": len(shared_events),
                "vol_minus_tpp_mean": diff_ci}
    else:
        # Exact or approximate matched trades
        tpp_dirs = np.where(tpp_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        tpp_epnl = compute_per_trade_pnl(tpp_signals, tpp_dirs, regime)
        vol_dirs = np.where(vol_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        vol_epnl = compute_per_trade_pnl(vol_signals, vol_dirs, regime)

        # For exact matches, use the join result — deduplicate to ensure 1:1
        match_keys = matched.select("event_id", "timestamp_ms", "dt_seconds").unique()
        tpp_matched = tpp_signals.join(
            match_keys,
            on=["event_id", "timestamp_ms", "dt_seconds"],
            how="inner",
        ).unique(subset=["event_id", "timestamp_ms", "dt_seconds"])
        vol_matched = vol_signals.join(
            match_keys,
            on=["event_id", "timestamp_ms", "dt_seconds"],
            how="inner",
        ).unique(subset=["event_id", "timestamp_ms", "dt_seconds"])

        # Align on same keys in same order
        sort_cols = ["event_id", "timestamp_ms", "dt_seconds"]
        tpp_matched = tpp_matched.sort(sort_cols)
        vol_matched = vol_matched.sort(sort_cols)

        # Ensure exact alignment
        common = tpp_matched.select(sort_cols).join(
            vol_matched.select(sort_cols), on=sort_cols, how="inner"
        )
        tpp_matched = tpp_matched.join(common, on=sort_cols, how="inner").sort(sort_cols)
        vol_matched = vol_matched.join(common, on=sort_cols, how="inner").sort(sort_cols)

        tpp_m_dirs = np.where(tpp_matched["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        vol_m_dirs = np.where(vol_matched["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
        tpp_m_epnl = compute_per_trade_pnl(tpp_matched, tpp_m_dirs, regime)
        vol_m_epnl = compute_per_trade_pnl(vol_matched, vol_m_dirs, regime)
        n_matched = len(tpp_m_epnl)

        diff = vol_m_epnl - tpp_m_epnl
        diff_ci = bootstrap_ci(diff, np.mean, rng=rng)
        print(f"\n  Matched trade comparison ({n_matched} trades):")
        print(f"    TPP mean: {tpp_m_epnl.mean():.4f}")
        print(f"    Volume mean: {vol_m_epnl.mean():.4f}")
        print(f"    Diff (Vol-TPP): {diff_ci[0]:>8.4f}  [{diff_ci[1]:>8.4f}, {diff_ci[2]:>8.4f}]")

        agree = (tpp_m_dirs == vol_m_dirs).sum()
        print(f"    Direction agreement: {agree}/{n_matched} ({agree/n_matched:.1%})")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(tpp_m_epnl, vol_m_epnl, alpha=0.4, s=20, color="steelblue")
        lim = max(abs(tpp_m_epnl).max(), abs(vol_m_epnl).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="TPP = Volume")
        ax.set_xlabel("TPP expected P&L ($)")
        ax.set_ylabel("Volume expected P&L ($)")
        ax.set_title(f"Matched Trade Comparison (n={n_matched})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "02_matched_scatter.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / '02_matched_scatter.png'}")

        return {"n_matched": n_matched, "diff_ci": diff_ci}


# ---------------------------------------------------------------------------
# Section 3: Stratification by market + time-of-day
# ---------------------------------------------------------------------------

def section_stratification(
    tpp_signals: pl.DataFrame,
    vol_signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 3: STRATIFICATION BY MARKET + TIME-OF-DAY")
    print("=" * 80)

    regime = "moderate"

    # Compute per-trade P&L
    tpp_dirs = np.where(tpp_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    tpp_epnl = compute_per_trade_pnl(tpp_signals, tpp_dirs, regime)
    vol_dirs = np.where(vol_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    vol_epnl = compute_per_trade_pnl(vol_signals, vol_dirs, regime)

    # Add P&L columns to signals for grouping
    tpp = tpp_signals.with_columns(pl.Series("expected_pnl", tpp_epnl))
    vol = vol_signals.with_columns(pl.Series("expected_pnl", vol_epnl))

    # --- 3a: By city ---
    print("\n  --- By City ---")
    all_cities = sorted(set(tpp["city"].unique().to_list()) | set(vol["city"].unique().to_list()))

    city_stats = []
    for city in all_cities:
        row = {"city": city}
        for label, df in [("TPP", tpp), ("Volume", vol)]:
            sub = df.filter(pl.col("city") == city)
            if sub.height < 10:
                row[f"{label}_n"] = sub.height
                row[f"{label}_mean"] = float("nan")
                row[f"{label}_ci_lo"] = float("nan")
                row[f"{label}_ci_hi"] = float("nan")
                continue
            epnl_vals = sub["expected_pnl"].to_numpy()
            ci = bootstrap_ci(epnl_vals, np.mean, rng=rng)
            row[f"{label}_n"] = sub.height
            row[f"{label}_mean"] = ci[0]
            row[f"{label}_ci_lo"] = ci[1]
            row[f"{label}_ci_hi"] = ci[2]
        city_stats.append(row)

        tpp_str = (f"TPP: {row.get('TPP_mean', float('nan')):>8.4f}" if not np.isnan(row.get("TPP_mean", float("nan"))) else "TPP: n/a")
        vol_str = (f"Vol: {row.get('Volume_mean', float('nan')):>8.4f}" if not np.isnan(row.get("Volume_mean", float("nan"))) else "Vol: n/a")
        print(f"    {city:<15s} {tpp_str}  {vol_str}")

    # Plot by city
    valid_cities = [c for c in city_stats
                    if not np.isnan(c.get("TPP_mean", float("nan")))
                    and not np.isnan(c.get("Volume_mean", float("nan")))]
    valid_cities.sort(key=lambda x: x["TPP_mean"])

    if valid_cities:
        fig, ax = plt.subplots(figsize=(12, max(6, len(valid_cities) * 0.5)))
        y_pos = np.arange(len(valid_cities))
        bar_height = 0.35

        tpp_means = [c["TPP_mean"] for c in valid_cities]
        tpp_errs_lo = [c["TPP_mean"] - c["TPP_ci_lo"] for c in valid_cities]
        tpp_errs_hi = [c["TPP_ci_hi"] - c["TPP_mean"] for c in valid_cities]

        vol_means = [c["Volume_mean"] for c in valid_cities]
        vol_errs_lo = [c["Volume_mean"] - c["Volume_ci_lo"] for c in valid_cities]
        vol_errs_hi = [c["Volume_ci_hi"] - c["Volume_mean"] for c in valid_cities]

        ax.barh(y_pos - bar_height/2, tpp_means, bar_height,
                xerr=[tpp_errs_lo, tpp_errs_hi],
                color="steelblue", alpha=0.7, label="TPP", capsize=3)
        ax.barh(y_pos + bar_height/2, vol_means, bar_height,
                xerr=[vol_errs_lo, vol_errs_hi],
                color="coral", alpha=0.7, label="Volume", capsize=3)

        labels = [f"{c['city']} (T:{c['TPP_n']}, V:{c['Volume_n']})" for c in valid_cities]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean Expected P&L per Trade ($)")
        ax.set_title("TPP vs Volume by City (Moderate Regime, 95% CI)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "03a_by_city.png", bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out_dir / '03a_by_city.png'}")

    # --- 3b: By time-of-day (hour UTC) ---
    print("\n  --- By Time-of-Day (Hour UTC) ---")
    tpp_hours = tpp.with_columns(
        ((pl.col("timestamp_ms") / 1000) % 86400 / 3600).floor().cast(pl.Int32).alias("hour_utc")
    )
    vol_hours = vol.with_columns(
        ((pl.col("timestamp_ms") / 1000) % 86400 / 3600).floor().cast(pl.Int32).alias("hour_utc")
    )

    hour_stats = []
    for hour in range(24):
        row = {"hour": hour}
        for label, df in [("TPP", tpp_hours), ("Volume", vol_hours)]:
            sub = df.filter(pl.col("hour_utc") == hour)
            if sub.height < 10:
                row[f"{label}_n"] = sub.height
                row[f"{label}_mean"] = float("nan")
                row[f"{label}_ci_lo"] = float("nan")
                row[f"{label}_ci_hi"] = float("nan")
                continue
            epnl_vals = sub["expected_pnl"].to_numpy()
            ci = bootstrap_ci(epnl_vals, np.mean, rng=rng)
            row[f"{label}_n"] = sub.height
            row[f"{label}_mean"] = ci[0]
            row[f"{label}_ci_lo"] = ci[1]
            row[f"{label}_ci_hi"] = ci[2]
        hour_stats.append(row)

    # Plot by hour
    valid_hours = [h for h in hour_stats
                   if not np.isnan(h.get("TPP_mean", float("nan")))
                   or not np.isnan(h.get("Volume_mean", float("nan")))]

    if valid_hours:
        fig, ax = plt.subplots(figsize=(14, 6))
        hours_x = [h["hour"] for h in valid_hours]
        bar_width = 0.35

        tpp_y = [h.get("TPP_mean", 0) if not np.isnan(h.get("TPP_mean", float("nan"))) else 0 for h in valid_hours]
        vol_y = [h.get("Volume_mean", 0) if not np.isnan(h.get("Volume_mean", float("nan"))) else 0 for h in valid_hours]

        x = np.arange(len(hours_x))
        ax.bar(x - bar_width/2, tpp_y, bar_width, color="steelblue", alpha=0.7, label="TPP")
        ax.bar(x + bar_width/2, vol_y, bar_width, color="coral", alpha=0.7, label="Volume")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}:00" for h in hours_x], rotation=45, fontsize=8)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean Expected P&L per Trade ($)")
        ax.set_title("TPP vs Volume by Time-of-Day (Hour UTC, Moderate Regime)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "03b_by_hour.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / '03b_by_hour.png'}")

    # Print summary
    for h in hour_stats:
        tpp_str = f"TPP: {h['TPP_mean']:>8.4f}" if not np.isnan(h.get("TPP_mean", float("nan"))) else "TPP:      n/a"
        vol_str = f"Vol: {h['Volume_mean']:>8.4f}" if not np.isnan(h.get("Volume_mean", float("nan"))) else "Vol:      n/a"
        print(f"    {h['hour']:02d}:00  {tpp_str} (n={h.get('TPP_n', 0):>3d})  "
              f"{vol_str} (n={h.get('Volume_n', 0):>3d})")


# ---------------------------------------------------------------------------
# Section 4: Cumulative P&L curves
# ---------------------------------------------------------------------------

def section_cumulative_pnl(
    tpp_signals: pl.DataFrame,
    vol_signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 4: CUMULATIVE P&L CURVES")
    print("=" * 80)

    regime = "moderate"

    tpp_dirs = np.where(tpp_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    tpp_epnl = compute_per_trade_pnl(tpp_signals, tpp_dirs, regime)
    vol_dirs = np.where(vol_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    vol_epnl = compute_per_trade_pnl(vol_signals, vol_dirs, regime)

    # Sort by timestamp for chronological cumulative curves
    tpp_ts = tpp_signals["timestamp_ms"].to_numpy()
    vol_ts = vol_signals["timestamp_ms"].to_numpy()

    tpp_order = np.argsort(tpp_ts)
    vol_order = np.argsort(vol_ts)

    tpp_cum = np.cumsum(tpp_epnl[tpp_order])
    vol_cum = np.cumsum(vol_epnl[vol_order])

    # Random baseline for reference
    random_dirs = rng.choice([-1.0, 1.0], size=len(tpp_signals))
    random_epnl = compute_per_trade_pnl(tpp_signals, random_dirs, regime)
    random_cum = np.cumsum(random_epnl[tpp_order])

    # Convert timestamps to datetime for x-axis
    tpp_dates = [datetime.datetime.fromtimestamp(t / 1000, tz=datetime.timezone.utc)
                 for t in tpp_ts[tpp_order]]
    vol_dates = [datetime.datetime.fromtimestamp(t / 1000, tz=datetime.timezone.utc)
                 for t in vol_ts[vol_order]]

    # --- 4a: Cumulative P&L over time ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(tpp_dates, tpp_cum, color="steelblue", linewidth=1.5, alpha=0.8, label=f"TPP (n={len(tpp_signals)})")
    ax.plot(vol_dates, vol_cum, color="coral", linewidth=1.5, alpha=0.8, label=f"Volume (n={len(vol_signals)})")
    ax.plot(tpp_dates, random_cum, color="gray", linewidth=1, alpha=0.5, label="Random baseline")
    ax.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Cumulative Expected P&L ($)")
    ax.set_title("Cumulative P&L: TPP vs Volume Baseline (Moderate Regime)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "04a_cumulative_pnl.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '04a_cumulative_pnl.png'}")

    # --- 4b: By trade index (equal-weight comparison) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(tpp_cum)), tpp_cum, color="steelblue", linewidth=1.5, alpha=0.8, label="TPP")
    ax.plot(range(len(vol_cum)), vol_cum, color="coral", linewidth=1.5, alpha=0.8, label="Volume")
    ax.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("Trade Number (chronological)")
    ax.set_ylabel("Cumulative Expected P&L ($)")
    ax.set_title("Cumulative P&L by Trade Number")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04b_cumulative_by_trade.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '04b_cumulative_by_trade.png'}")

    # --- 4c: Per dt_seconds breakdown ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, dt_s in enumerate(DT_VALUES):
        ax = axes[idx // 2][idx % 2]

        tpp_dt = tpp_signals.filter(pl.col("dt_seconds") == dt_s)
        vol_dt = vol_signals.filter(pl.col("dt_seconds") == dt_s)

        if tpp_dt.height > 0:
            tpp_d = np.where(tpp_dt["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
            tpp_e = compute_per_trade_pnl(tpp_dt, tpp_d, regime)
            tpp_t = tpp_dt["timestamp_ms"].to_numpy()
            order = np.argsort(tpp_t)
            ax.plot(range(len(order)), np.cumsum(tpp_e[order]),
                    color="steelblue", linewidth=1.2, alpha=0.8, label=f"TPP (n={tpp_dt.height})")

        if vol_dt.height > 0:
            vol_d = np.where(vol_dt["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
            vol_e = compute_per_trade_pnl(vol_dt, vol_d, regime)
            vol_t = vol_dt["timestamp_ms"].to_numpy()
            order = np.argsort(vol_t)
            ax.plot(range(len(order)), np.cumsum(vol_e[order]),
                    color="coral", linewidth=1.2, alpha=0.8, label=f"Volume (n={vol_dt.height})")

        ax.axhline(0, color="black", linestyle="-", alpha=0.2)
        ax.set_title(f"dt = {dt_s}s")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative E[P&L] ($)")
        ax.legend(fontsize=8)

    fig.suptitle("Cumulative P&L by Look-Ahead Window", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "04c_cumulative_by_dt.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '04c_cumulative_by_dt.png'}")

    # Summary stats
    print(f"\n  TPP final cumulative P&L:    ${tpp_cum[-1]:>10.2f}")
    print(f"  Volume final cumulative P&L: ${vol_cum[-1]:>10.2f}")
    print(f"  Random baseline P&L:         ${random_cum[-1]:>10.2f}")

    tpp_max_dd = float(np.min(tpp_cum - np.maximum.accumulate(tpp_cum)))
    vol_max_dd = float(np.min(vol_cum - np.maximum.accumulate(vol_cum)))
    print(f"\n  TPP max drawdown:    ${tpp_max_dd:>10.2f}")
    print(f"  Volume max drawdown: ${vol_max_dd:>10.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TPP vs Volume baseline comparison (pm-kcw.2)"
    )
    parser.add_argument("--signals", type=Path, default=Path("data/signals/signals.parquet"))
    parser.add_argument("--events-dir", type=Path, default=Path("data/events"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/tpp_vs_volume"))
    parser.add_argument("--out-parquet", type=Path, default=Path("data/backtest/tpp_vs_volume_comparison.parquet"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-vol-signals", type=int, default=485)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load TPP signals (best config: threshold_pct=10)
    # -----------------------------------------------------------------------
    print("Loading TPP signals...")
    tpp_all = pl.read_parquet(args.signals)
    tpp_signals = tpp_all.filter(pl.col("threshold_pct") == 10)
    print(f"  TPP: {tpp_signals.height} signals (threshold_pct=10), "
          f"{tpp_signals['event_id'].n_unique()} events")

    # -----------------------------------------------------------------------
    # Generate volume signals (60s window, matching TPP trade count)
    # -----------------------------------------------------------------------
    print(f"\nGenerating volume signals (window={VOLUME_WINDOW_S}s)...")

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
    print(f"  Found {len(holdout_eids)} held-out events")

    all_vol_signals = []
    t0 = time.time()
    for i, eid in enumerate(holdout_eids):
        sigs = generate_volume_signals_for_event(eid, args.events_dir)
        all_vol_signals.extend(sigs)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(holdout_eids)}] {len(all_vol_signals):,} candidates ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Total: {len(all_vol_signals):,} candidates ({elapsed:.1f}s)")

    if not all_vol_signals:
        print("ERROR: No volume signals generated.")
        return

    vol_df = pl.DataFrame(all_vol_signals)

    # Select top signals by volume_strength per dt_seconds
    print(f"\n  Selecting top {args.target_vol_signals} per dt window...")
    selected = []
    for dt_s in DT_VALUES:
        subset = vol_df.filter(pl.col("dt_seconds") == dt_s)
        n_take = min(args.target_vol_signals, subset.height)
        top = subset.sort("volume_strength", descending=True).head(n_take)
        selected.append(top)
        print(f"    dt={dt_s:>3}s: {top.height} signals")

    vol_signals = pl.concat(selected)
    print(f"  Volume: {vol_signals.height} signals, "
          f"{vol_signals['event_id'].n_unique()} events")

    # -----------------------------------------------------------------------
    # Run comparison sections
    # -----------------------------------------------------------------------
    results = {}

    results["aggregate"] = section_aggregate(tpp_signals, vol_signals, args.out_dir, rng)
    results["matched"] = section_matched(tpp_signals, vol_signals, args.out_dir, rng)
    section_stratification(tpp_signals, vol_signals, args.out_dir, rng)
    section_cumulative_pnl(tpp_signals, vol_signals, args.out_dir, rng)

    # -----------------------------------------------------------------------
    # Save comparison data
    # -----------------------------------------------------------------------
    # Build a combined per-trade parquet
    tpp_dirs = np.where(tpp_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    vol_dirs = np.where(vol_signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)

    regime = "moderate"
    tpp_epnl = compute_per_trade_pnl(tpp_signals, tpp_dirs, regime)
    vol_epnl = compute_per_trade_pnl(vol_signals, vol_dirs, regime)

    tpp_out = tpp_signals.select("event_id", "city", "timestamp_ms", "dt_seconds",
                                 "current_price", "price_change").with_columns(
        pl.lit("TPP").alias("strategy"),
        pl.Series("expected_pnl", tpp_epnl),
        pl.Series("direction", tpp_dirs),
    )
    vol_out = vol_signals.select("event_id", "city", "timestamp_ms", "dt_seconds",
                                 "current_price", "price_change").with_columns(
        pl.lit("Volume").alias("strategy"),
        pl.Series("expected_pnl", vol_epnl),
        pl.Series("direction", vol_dirs),
    )
    combined = pl.concat([tpp_out, vol_out])
    combined.write_parquet(args.out_parquet)
    print(f"\nSaved per-trade comparison: {args.out_parquet} ({combined.height} rows)")

    # Save summary JSON
    summary = {
        "tpp_n": int(tpp_signals.height),
        "vol_n": int(vol_signals.height),
        "tpp_config": {"threshold_pct": 10},
        "vol_config": {"volume_window_s": VOLUME_WINDOW_S},
        "regime": regime,
    }
    for strategy in ["TPP", "Volume"]:
        for metric in ["mean_pnl", "total_pnl", "sharpe", "hit_rate", "win_rate"]:
            if strategy in results.get("aggregate", {}):
                m = results["aggregate"][strategy].get(metric, {})
                if isinstance(m, dict):
                    summary[f"{strategy.lower()}_{metric}"] = m.get("point", None)
                    summary[f"{strategy.lower()}_{metric}_ci"] = [m.get("ci_lo"), m.get("ci_hi")]

    summary_path = args.out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
