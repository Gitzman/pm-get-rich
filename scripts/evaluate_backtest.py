"""Honest evaluation of TPP backtest with CI and adverse selection (pm-kxg.4).

Produces:
  1. Per-trade P&L with bootstrap confidence intervals
  2. Model vs baseline comparison (random, bucket-shuffled)
  3. Stratification by city and time-to-resolution
  4. Capacity analysis (P&L vs order size)
  5. Adverse selection diagnostic (post-fill price drift)

Figures saved to data/reports/backtest_eval/.
"""
from __future__ import annotations

import argparse
import json
import os
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
THETA = 0.050  # weather markets
N_BOOTSTRAP = 5000
SEED = 42

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
) -> pl.DataFrame:
    """Compute per-trade P&L for a set of signals with given directions."""
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
    fill_probs = np.array([
        fill_probability(taker_vol, resting_depth, contracts, assumptions)
        for _ in range(n)
    ])
    exit_prices = np.clip(prices + price_changes, 0.01, 0.99)
    exit_fees = np.array([
        taker_fee(float(ep), contracts, theta) for ep in exit_prices
    ])
    adv_sel = np.full(n, adverse_selection_cost(contracts, assumptions=assumptions))
    net_pnl = gross_pnl - exit_fees - adv_sel
    expected_pnl = fill_probs * net_pnl

    correct = (directions * price_changes) > 0
    flat = np.abs(price_changes) < 1e-6

    return pl.DataFrame({
        "event_id": df["event_id"],
        "city": df["city"],
        "timestamp_ms": df["timestamp_ms"],
        "dt_seconds": df["dt_seconds"],
        "threshold_pct": df["threshold_pct"],
        "current_price": prices,
        "price_change": price_changes,
        "direction": directions,
        "gross_pnl": gross_pnl,
        "fill_prob": fill_probs,
        "exit_fee": exit_fees,
        "adverse_selection": adv_sel,
        "net_pnl": net_pnl,
        "expected_pnl": expected_pnl,
        "correct_direction": correct,
        "flat_price": flat,
    })


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
# Event metadata loader
# ---------------------------------------------------------------------------

def load_event_metadata(events_dir: Path, event_ids: list[str]) -> pl.DataFrame:
    """Load city and time range from event _meta.json files."""
    rows = []
    for eid in event_ids:
        meta_path = events_dir / eid / "_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        tr = meta.get("time_range", {})
        rows.append({
            "event_id": str(eid),
            "event_city": meta.get("city", "unknown"),
            "event_start_s": tr.get("start_epoch_s", 0),
            "event_end_s": tr.get("end_epoch_s", 0),
            "event_n_trades": meta.get("n_trades", 0),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 1: Per-trade P&L with bootstrap CI
# ---------------------------------------------------------------------------

def section_per_trade_pnl(
    signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 1: PER-TRADE P&L WITH BOOTSTRAP CI")
    print("=" * 80)

    # Use moderate regime as the reference
    regime = "moderate"
    model_dirs = np.where(signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    trades = compute_per_trade_pnl(signals, model_dirs, regime)

    epnl = trades["expected_pnl"].to_numpy()

    # Bootstrap metrics
    metrics = {}
    for name, fn in [
        ("mean_expected_pnl", np.mean),
        ("median_expected_pnl", np.median),
        ("total_expected_pnl", np.sum),
        ("sharpe", sharpe_fn),
    ]:
        pt, lo, hi = bootstrap_ci(epnl, fn, rng=rng)
        metrics[name] = (pt, lo, hi)
        print(f"  {name:>25s}: {pt:>10.4f}  [{lo:>10.4f}, {hi:>10.4f}]")

    # Hit rate (excluding flat)
    non_flat = trades.filter(~pl.col("flat_price"))
    hit_vals = non_flat["correct_direction"].cast(pl.Float64).to_numpy()
    pt, lo, hi = bootstrap_ci(hit_vals, np.mean, rng=rng)
    metrics["hit_rate"] = (pt, lo, hi)
    print(f"  {'hit_rate':>25s}: {pt:>10.4f}  [{lo:>10.4f}, {hi:>10.4f}]")

    # Win rate
    win_vals = (epnl > 0).astype(float)
    pt, lo, hi = bootstrap_ci(win_vals, np.mean, rng=rng)
    metrics["win_rate"] = (pt, lo, hi)
    print(f"  {'win_rate':>25s}: {pt:>10.4f}  [{lo:>10.4f}, {hi:>10.4f}]")

    # P&L distribution figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(epnl, bins=80, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(epnl.mean(), color="gold", linestyle="-", linewidth=2, label=f"mean={epnl.mean():.4f}")
    ax.set_xlabel("Expected P&L per trade ($)")
    ax.set_ylabel("Count")
    ax.set_title("Per-trade Expected P&L Distribution (Moderate Regime)")
    ax.legend()

    # Bootstrap distribution of mean
    boot_means = np.array([
        np.mean(rng.choice(epnl, size=len(epnl), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    ax = axes[1]
    ax.hist(boot_means, bins=60, alpha=0.7, color="coral", edgecolor="white")
    ci = metrics["mean_expected_pnl"]
    ax.axvline(ci[1], color="black", linestyle="--", label=f"95% CI: [{ci[1]:.4f}, {ci[2]:.4f}]")
    ax.axvline(ci[2], color="black", linestyle="--")
    ax.axvline(0, color="red", linestyle="-", linewidth=2, alpha=0.5, label="zero")
    ax.set_xlabel("Bootstrap mean expected P&L")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap Distribution of Mean Expected P&L")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "01_per_trade_pnl.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '01_per_trade_pnl.png'}")

    return trades, metrics


# ---------------------------------------------------------------------------
# Section 2: Model vs baselines
# ---------------------------------------------------------------------------

def section_baselines(
    signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 2: MODEL VS BASELINES")
    print("=" * 80)

    regime = "moderate"
    bucket_pos = signals["pred_bucket_pos"].to_numpy()
    model_dirs = np.where(bucket_pos > 0.5, 1.0, -1.0)
    random_dirs = rng.choice([-1.0, 1.0], size=len(signals))

    # Shuffled: permute pred_bucket_pos within each event
    shuffled_bp = bucket_pos.copy()
    event_ids = signals["event_id"].to_numpy()
    for eid in np.unique(event_ids):
        mask = event_ids == eid
        shuffled_bp[mask] = rng.permutation(shuffled_bp[mask])
    shuffled_dirs = np.where(shuffled_bp > 0.5, 1.0, -1.0)

    results = {}
    for label, dirs in [("model", model_dirs), ("random", random_dirs), ("shuffled", shuffled_dirs)]:
        trades = compute_per_trade_pnl(signals, dirs, regime)
        epnl = trades["expected_pnl"].to_numpy()
        non_flat = trades.filter(~pl.col("flat_price"))
        hit_vals = non_flat["correct_direction"].cast(pl.Float64).to_numpy()

        mean_ci = bootstrap_ci(epnl, np.mean, rng=rng)
        total_ci = bootstrap_ci(epnl, np.sum, rng=rng)
        sharpe_ci = bootstrap_ci(epnl, sharpe_fn, rng=rng)
        hit_ci = bootstrap_ci(hit_vals, np.mean, rng=rng)

        results[label] = {
            "epnl": epnl,
            "mean_ci": mean_ci,
            "total_ci": total_ci,
            "sharpe_ci": sharpe_ci,
            "hit_ci": hit_ci,
        }
        print(f"\n  {label:>10s}:")
        print(f"    mean_expected_pnl: {mean_ci[0]:>10.4f}  [{mean_ci[1]:>10.4f}, {mean_ci[2]:>10.4f}]")
        print(f"    total_expected_pnl: {total_ci[0]:>10.2f}  [{total_ci[1]:>10.2f}, {total_ci[2]:>10.2f}]")
        print(f"    sharpe:            {sharpe_ci[0]:>10.4f}  [{sharpe_ci[1]:>10.4f}, {sharpe_ci[2]:>10.4f}]")
        print(f"    hit_rate:          {hit_ci[0]:>10.4f}  [{hit_ci[1]:>10.4f}, {hit_ci[2]:>10.4f}]")

    # CI overlap check
    print("\n  --- CI Overlap Analysis ---")
    for baseline in ["random", "shuffled"]:
        m_lo, m_hi = results["model"]["mean_ci"][1], results["model"]["mean_ci"][2]
        b_lo, b_hi = results[baseline]["mean_ci"][1], results[baseline]["mean_ci"][2]
        overlap = max(0, min(m_hi, b_hi) - max(m_lo, b_lo))
        model_width = m_hi - m_lo
        if model_width > 0:
            overlap_frac = overlap / model_width
        else:
            overlap_frac = 1.0
        sep = m_lo > b_hi or b_lo > m_hi
        print(f"  model vs {baseline}: overlap={overlap_frac:.1%}, "
              f"CIs {'SEPARATED' if sep else 'OVERLAP'}")

    # Figure: side-by-side CI comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric_key, metric_label in [
        (axes[0], "mean_ci", "Mean Expected P&L ($)"),
        (axes[1], "sharpe_ci", "Sharpe Ratio"),
        (axes[2], "hit_ci", "Hit Rate"),
    ]:
        labels = []
        points = []
        lows = []
        highs = []
        for sig_type in ["model", "random", "shuffled"]:
            ci = results[sig_type][metric_key]
            labels.append(sig_type)
            points.append(ci[0])
            lows.append(ci[0] - ci[1])
            highs.append(ci[2] - ci[0])

        colors = ["steelblue", "gray", "orange"]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, points, xerr=[lows, highs], color=colors, alpha=0.7,
                edgecolor="white", capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel(metric_label)
        ax.set_title(metric_label)

    fig.suptitle("Model vs Baselines (Moderate Fill Regime, 95% Bootstrap CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "02_baselines.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '02_baselines.png'}")

    # Across all regimes
    print("\n  --- All Fill Regimes ---")
    regime_summary = []
    for reg_name in FILL_REGIMES:
        for label, dirs in [("model", model_dirs), ("random", random_dirs), ("shuffled", shuffled_dirs)]:
            trades = compute_per_trade_pnl(signals, dirs, reg_name)
            epnl = trades["expected_pnl"].to_numpy()
            mean_ci = bootstrap_ci(epnl, np.mean, rng=rng)
            regime_summary.append({
                "regime": reg_name,
                "signal_type": label,
                "mean_pnl": mean_ci[0],
                "ci_lo": mean_ci[1],
                "ci_hi": mean_ci[2],
            })
            print(f"    {reg_name:<14s} {label:<10s}: "
                  f"mean={mean_ci[0]:>8.4f}  [{mean_ci[1]:>8.4f}, {mean_ci[2]:>8.4f}]")

    return results


# ---------------------------------------------------------------------------
# Section 3: Stratification
# ---------------------------------------------------------------------------

def section_stratification(
    signals: pl.DataFrame,
    events_dir: Path,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 3: STRATIFICATION")
    print("=" * 80)

    regime = "moderate"
    model_dirs = np.where(signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)
    trades = compute_per_trade_pnl(signals, model_dirs, regime)

    # --- 3a: By city ---
    print("\n  --- By City ---")
    cities = trades["city"].unique().sort().to_list()
    city_stats = []
    for city in cities:
        mask = trades["city"] == city
        epnl = trades.filter(mask)["expected_pnl"].to_numpy()
        if len(epnl) < 10:
            continue
        mean_ci = bootstrap_ci(epnl, np.mean, rng=rng)
        city_stats.append({
            "city": city,
            "n": len(epnl),
            "mean_pnl": mean_ci[0],
            "ci_lo": mean_ci[1],
            "ci_hi": mean_ci[2],
        })
        print(f"    {city:<15s} n={len(epnl):>4d}  "
              f"mean={mean_ci[0]:>8.4f}  [{mean_ci[1]:>8.4f}, {mean_ci[2]:>8.4f}]")

    # Sort by mean P&L for plotting
    city_stats.sort(key=lambda x: x["mean_pnl"])

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(city_stats))
    means = [c["mean_pnl"] for c in city_stats]
    lows = [c["mean_pnl"] - c["ci_lo"] for c in city_stats]
    highs = [c["ci_hi"] - c["mean_pnl"] for c in city_stats]
    labels = [f"{c['city']} (n={c['n']})" for c in city_stats]
    colors = ["steelblue" if m > 0 else "salmon" for m in means]

    ax.barh(y_pos, means, xerr=[lows, highs], color=colors, alpha=0.7,
            edgecolor="white", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Expected P&L per Trade ($)")
    ax.set_title("Mean Expected P&L by City (Moderate Regime, 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "03a_by_city.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '03a_by_city.png'}")

    # --- 3b: By time-to-resolution ---
    print("\n  --- By Time-to-Resolution ---")
    event_ids = signals["event_id"].unique().sort().to_list()
    event_meta = load_event_metadata(events_dir, event_ids)

    # Join time-to-resolution
    trades_with_meta = trades.join(event_meta, on="event_id", how="left")
    # Compute hours to resolution
    trades_with_meta = trades_with_meta.with_columns(
        ((pl.col("event_end_s") - pl.col("timestamp_ms") / 1000) / 3600).alias("hours_to_resolution")
    )

    # Bucket by hours to resolution
    h2r = trades_with_meta["hours_to_resolution"].to_numpy()
    epnl_all = trades_with_meta["expected_pnl"].to_numpy()
    bins = [0, 6, 12, 24, 48, 200]
    bin_labels = ["0-6h", "6-12h", "12-24h", "24-48h", "48h+"]
    ttr_stats = []
    for i in range(len(bins) - 1):
        mask = (h2r >= bins[i]) & (h2r < bins[i + 1])
        epnl_bin = epnl_all[mask]
        if len(epnl_bin) < 10:
            continue
        mean_ci = bootstrap_ci(epnl_bin, np.mean, rng=rng)
        ttr_stats.append({
            "bin": bin_labels[i],
            "n": int(mask.sum()),
            "mean_pnl": mean_ci[0],
            "ci_lo": mean_ci[1],
            "ci_hi": mean_ci[2],
        })
        print(f"    {bin_labels[i]:<10s} n={int(mask.sum()):>4d}  "
              f"mean={mean_ci[0]:>8.4f}  [{mean_ci[1]:>8.4f}, {mean_ci[2]:>8.4f}]")

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(ttr_stats))
    means = [s["mean_pnl"] for s in ttr_stats]
    lows = [s["mean_pnl"] - s["ci_lo"] for s in ttr_stats]
    highs = [s["ci_hi"] - s["mean_pnl"] for s in ttr_stats]
    labels = [f"{s['bin']}\n(n={s['n']})" for s in ttr_stats]
    colors = ["steelblue" if m > 0 else "salmon" for m in means]

    ax.bar(x_pos, means, yerr=[lows, highs], color=colors, alpha=0.7,
           edgecolor="white", capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Expected P&L per Trade ($)")
    ax.set_title("Mean Expected P&L by Time-to-Resolution (Moderate Regime, 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "03b_by_time_to_resolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '03b_by_time_to_resolution.png'}")

    # --- 3c: By dt_seconds and threshold_pct ---
    print("\n  --- By dt_seconds x threshold_pct ---")
    dt_thr_stats = []
    for dt in sorted(trades["dt_seconds"].unique().to_list()):
        for thr in sorted(trades["threshold_pct"].unique().to_list()):
            mask = (trades["dt_seconds"] == dt) & (trades["threshold_pct"] == thr)
            epnl = trades.filter(mask)["expected_pnl"].to_numpy()
            if len(epnl) < 10:
                continue
            mean_ci = bootstrap_ci(epnl, np.mean, rng=rng)
            hit_vals = trades.filter(mask & ~pl.col("flat_price"))["correct_direction"].cast(pl.Float64).to_numpy()
            hit_ci = bootstrap_ci(hit_vals, np.mean, rng=rng) if len(hit_vals) > 10 else (0, 0, 0)
            dt_thr_stats.append({
                "dt": dt, "thr": thr, "n": len(epnl),
                "mean_pnl": mean_ci[0], "ci_lo": mean_ci[1], "ci_hi": mean_ci[2],
                "hit": hit_ci[0], "hit_lo": hit_ci[1], "hit_hi": hit_ci[2],
            })
            print(f"    dt={dt:>3d}s thr={thr:>2d}%  n={len(epnl):>4d}  "
                  f"mean_pnl={mean_ci[0]:>8.4f}  [{mean_ci[1]:>8.4f}, {mean_ci[2]:>8.4f}]  "
                  f"hit={hit_ci[0]:.3f}")

    # Heatmap of mean P&L
    dt_vals = sorted(set(s["dt"] for s in dt_thr_stats))
    thr_vals = sorted(set(s["thr"] for s in dt_thr_stats))
    heatmap = np.zeros((len(dt_vals), len(thr_vals)))
    for s in dt_thr_stats:
        i = dt_vals.index(s["dt"])
        j = thr_vals.index(s["thr"])
        heatmap[i, j] = s["mean_pnl"]

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(heatmap.min()), abs(heatmap.max()))
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(thr_vals)))
    ax.set_xticklabels([f"{t}%" for t in thr_vals])
    ax.set_yticks(range(len(dt_vals)))
    ax.set_yticklabels([f"{d}s" for d in dt_vals])
    ax.set_xlabel("Threshold %")
    ax.set_ylabel("dt_seconds")
    ax.set_title("Mean Expected P&L by (dt, threshold)")
    for i in range(len(dt_vals)):
        for j in range(len(thr_vals)):
            ax.text(j, i, f"{heatmap[i,j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean Expected P&L ($)")
    fig.tight_layout()
    fig.savefig(out_dir / "03c_dt_threshold_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '03c_dt_threshold_heatmap.png'}")

    return trades_with_meta


# ---------------------------------------------------------------------------
# Section 4: Capacity analysis
# ---------------------------------------------------------------------------

def section_capacity(
    signals: pl.DataFrame,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 4: CAPACITY ANALYSIS")
    print("=" * 80)

    model_dirs = np.where(signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)

    contract_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    regime = "moderate"
    r = FILL_REGIMES[regime]

    print(f"\n  Regime: {regime} (taker_vol={r['taker_volume']}, "
          f"resting_depth={r['resting_depth']})")
    print(f"  {'Contracts':>10s}  {'Fill Prob':>10s}  {'Mean Net P&L':>14s}  "
          f"{'Mean Exp P&L':>14s}  {'Total Exp':>12s}")

    cap_results = []
    for c in contract_sizes:
        trades = compute_per_trade_pnl(signals, model_dirs, regime, contracts=c)
        epnl = trades["expected_pnl"].to_numpy()
        fp = trades["fill_prob"].to_numpy()[0]  # same for all trades in regime
        mean_net = trades["net_pnl"].to_numpy().mean()
        mean_exp = epnl.mean()
        total_exp = epnl.sum()

        cap_results.append({
            "contracts": c,
            "fill_prob": fp,
            "mean_net_pnl": mean_net,
            "mean_expected_pnl": mean_exp,
            "total_expected_pnl": total_exp,
        })
        print(f"  {c:>10d}  {fp:>10.4f}  {mean_net:>14.4f}  "
              f"{mean_exp:>14.4f}  {total_exp:>12.2f}")

    # Figure: capacity curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cs = [r["contracts"] for r in cap_results]

    ax = axes[0]
    ax.plot(cs, [r["fill_prob"] for r in cap_results], "o-", color="steelblue")
    ax.set_xlabel("Order Size (contracts)")
    ax.set_ylabel("Fill Probability")
    ax.set_title("Fill Probability vs Order Size")
    ax.set_xscale("log")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% fill")
    ax.legend()

    ax = axes[1]
    ax.plot(cs, [r["mean_expected_pnl"] for r in cap_results], "o-", color="steelblue")
    ax.set_xlabel("Order Size (contracts)")
    ax.set_ylabel("Mean Expected P&L ($)")
    ax.set_title("Mean Expected P&L vs Order Size")
    ax.set_xscale("log")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)

    ax = axes[2]
    ax.plot(cs, [r["total_expected_pnl"] for r in cap_results], "o-", color="steelblue")
    ax.set_xlabel("Order Size (contracts)")
    ax.set_ylabel("Total Expected P&L ($)")
    ax.set_title("Total Expected P&L vs Order Size")
    ax.set_xscale("log")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle("Capacity Analysis (Moderate Fill Regime)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "04_capacity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '04_capacity.png'}")


# ---------------------------------------------------------------------------
# Section 5: Adverse selection diagnostic
# ---------------------------------------------------------------------------

def section_adverse_selection(
    signals: pl.DataFrame,
    events_dir: Path,
    out_dir: Path,
    rng: np.random.Generator,
):
    print("\n" + "=" * 80)
    print("SECTION 5: ADVERSE SELECTION DIAGNOSTIC")
    print("=" * 80)

    model_dirs = np.where(signals["pred_bucket_pos"].to_numpy() > 0.5, 1.0, -1.0)

    # For adverse selection, we measure post-signal price drift using
    # the event-level trade data. For each signal, compute price change
    # over 30s and 60s windows after the signal timestamp.
    print("\n  Computing post-signal price drift from event data...")

    drift_windows = [30, 60, 120]
    drift_records = []

    # Group signals by event_id for efficient event data loading
    sig_by_event = signals.group_by("event_id").agg(pl.all())

    event_ids = sig_by_event["event_id"].to_list()
    for eid in event_ids:
        event_path = events_dir / eid / "events.parquet"
        if not event_path.exists():
            continue

        event_trades = pl.read_parquet(event_path)
        event_signals = signals.filter(pl.col("event_id") == eid)

        for row in event_signals.iter_rows(named=True):
            ts_ms = row["timestamp_ms"]
            direction = 1.0 if row["pred_bucket_pos"] > 0.5 else -1.0
            current_suit = row["current_suit"]

            # Get trades for this market (suit) after signal time
            future_trades = event_trades.filter(
                (pl.col("timestamp_ms") > ts_ms) &
                (pl.col("suit") == current_suit)
            ).sort("timestamp_ms")

            if len(future_trades) == 0:
                continue

            signal_price = row["current_price"]

            for window_s in drift_windows:
                window_trades = future_trades.filter(
                    pl.col("timestamp_ms") <= ts_ms + window_s * 1000
                )
                if len(window_trades) == 0:
                    continue

                # Volume-weighted average price in window
                prices = window_trades["price"].to_numpy()
                sizes = window_trades["size"].to_numpy()
                if sizes.sum() > 0:
                    vwap = float(np.average(prices, weights=sizes))
                else:
                    vwap = float(prices.mean())

                # Price drift: how much did the price move?
                raw_drift = vwap - signal_price
                # Directional drift: positive = moved against our position
                adverse_drift = -direction * raw_drift

                drift_records.append({
                    "event_id": eid,
                    "city": row["city"],
                    "dt_seconds": row["dt_seconds"],
                    "threshold_pct": row["threshold_pct"],
                    "direction": direction,
                    "signal_price": signal_price,
                    "window_s": window_s,
                    "vwap": vwap,
                    "raw_drift": raw_drift,
                    "adverse_drift": adverse_drift,
                    "window_volume": float(sizes.sum()),
                })

    if not drift_records:
        print("  WARNING: No post-signal trades found for drift analysis.")
        return

    drift_df = pl.DataFrame(drift_records)
    print(f"  Computed {len(drift_df):,} drift observations")

    # Summary by window
    print("\n  --- Post-Signal Price Drift ---")
    for w in drift_windows:
        sub = drift_df.filter(pl.col("window_s") == w)
        if len(sub) == 0:
            continue
        ad = sub["adverse_drift"].to_numpy()
        mean_ci = bootstrap_ci(ad, np.mean, rng=rng)
        median_ci = bootstrap_ci(ad, np.median, rng=rng)
        pct_adverse = float((ad > 0).sum()) / len(ad)
        print(f"    {w:>3d}s window: n={len(ad):>5d}  "
              f"mean_adverse={mean_ci[0]:>8.5f}  [{mean_ci[1]:>8.5f}, {mean_ci[2]:>8.5f}]  "
              f"median={median_ci[0]:>8.5f}  pct_adverse={pct_adverse:.1%}")

    # Compare to assumed adverse selection cost
    default_adv = 0.5 * 0.01  # 0.5 ticks × $0.01/tick = $0.005 per contract
    print(f"\n  Assumed adverse selection: {default_adv:.4f} per contract")

    for w in drift_windows:
        sub = drift_df.filter(pl.col("window_s") == w)
        if len(sub) == 0:
            continue
        actual = float(sub["adverse_drift"].mean())
        ratio = actual / default_adv if default_adv > 0 else float("inf")
        print(f"    {w:>3d}s actual mean drift: {actual:>8.5f}  ({ratio:.1f}x assumed)")

    # Figure: drift distributions
    fig, axes = plt.subplots(1, len(drift_windows), figsize=(5 * len(drift_windows), 5))
    if len(drift_windows) == 1:
        axes = [axes]

    for ax, w in zip(axes, drift_windows):
        sub = drift_df.filter(pl.col("window_s") == w)
        if len(sub) == 0:
            continue
        ad = sub["adverse_drift"].to_numpy()
        # Clip for display
        clip_lo, clip_hi = np.percentile(ad, [1, 99])
        ad_clipped = ad[(ad >= clip_lo) & (ad <= clip_hi)]

        ax.hist(ad_clipped, bins=60, alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="zero")
        ax.axvline(ad.mean(), color="gold", linestyle="-", linewidth=2,
                   label=f"mean={ad.mean():.5f}")
        ax.axvline(default_adv, color="green", linestyle="--", linewidth=2,
                   label=f"assumed={default_adv:.4f}")
        ax.set_xlabel("Adverse Drift ($ per contract)")
        ax.set_ylabel("Count")
        ax.set_title(f"Post-Signal Drift ({w}s window)")
        ax.legend(fontsize=7)

    fig.suptitle("Adverse Selection Diagnostic: Post-Fill Price Drift", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "05_adverse_selection.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_dir / '05_adverse_selection.png'}")

    # Adverse selection by city
    print("\n  --- Adverse Selection by City (60s window) ---")
    sub60 = drift_df.filter(pl.col("window_s") == 60)
    if len(sub60) > 0:
        for city in sorted(sub60["city"].unique().to_list()):
            city_drift = sub60.filter(pl.col("city") == city)["adverse_drift"].to_numpy()
            if len(city_drift) < 10:
                continue
            mean_ci = bootstrap_ci(city_drift, np.mean, rng=rng)
            print(f"    {city:<15s} n={len(city_drift):>4d}  "
                  f"mean_adverse={mean_ci[0]:>8.5f}  [{mean_ci[1]:>8.5f}, {mean_ci[2]:>8.5f}]")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def section_verdict(metrics: dict, baseline_results: dict):
    print("\n" + "=" * 80)
    print("VERDICT: HONEST ASSESSMENT")
    print("=" * 80)

    mean_pnl = metrics["mean_expected_pnl"]
    sharpe = metrics.get("sharpe", (0, 0, 0))
    hit = metrics.get("hit_rate", (0, 0, 0))

    # Check if mean P&L CI excludes zero
    pnl_sig = mean_pnl[1] > 0 or mean_pnl[2] < 0
    pnl_positive = mean_pnl[0] > 0

    # Check if model beats baselines
    model_mean = baseline_results["model"]["mean_ci"]
    random_mean = baseline_results["random"]["mean_ci"]
    shuffled_mean = baseline_results["shuffled"]["mean_ci"]

    beats_random = model_mean[1] > random_mean[2]  # model CI_lo > random CI_hi
    beats_shuffled = model_mean[1] > shuffled_mean[2]

    print(f"\n  Mean expected P&L:  {mean_pnl[0]:>8.4f}  [{mean_pnl[1]:>8.4f}, {mean_pnl[2]:>8.4f}]")
    print(f"  Sharpe ratio:       {sharpe[0]:>8.4f}  [{sharpe[1]:>8.4f}, {sharpe[2]:>8.4f}]")
    print(f"  Hit rate:           {hit[0]:>8.4f}  [{hit[1]:>8.4f}, {hit[2]:>8.4f}]")
    print()
    print(f"  P&L significantly != 0:       {'YES' if pnl_sig else 'NO'}")
    print(f"  P&L positive:                 {'YES' if pnl_positive else 'NO'}")
    print(f"  Beats random (CI separated):  {'YES' if beats_random else 'NO'}")
    print(f"  Beats shuffled (CI separated):{'YES' if beats_shuffled else 'NO'}")
    print()

    if pnl_positive and pnl_sig and beats_random and beats_shuffled:
        verdict = "POSITIVE — Model shows statistically significant edge over baselines."
    elif pnl_positive and (beats_random or beats_shuffled):
        verdict = "WEAK POSITIVE — Model shows positive P&L but CIs overlap with at least one baseline."
    elif pnl_positive:
        verdict = "INCONCLUSIVE — Model P&L is positive but not distinguishable from baselines."
    else:
        verdict = "NEGATIVE — Model does not show positive expected P&L after costs."

    print(f"  VERDICT: {verdict}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Honest evaluation of TPP backtest (pm-kxg.4)"
    )
    parser.add_argument(
        "--signals", type=Path, default=Path("data/signals/signals.parquet")
    )
    parser.add_argument(
        "--events-dir", type=Path, default=Path("data/events")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/reports/backtest_eval")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BACKTEST EVALUATION: TPP Signal → Cost-Adjusted P&L")
    print(f"Signals: {args.signals}")
    print(f"Events:  {args.events_dir}")
    print(f"Output:  {args.out_dir}")
    print("=" * 80)

    # Load signals
    signals = pl.read_parquet(args.signals)
    print(f"\nLoaded {len(signals):,} signals, "
          f"{signals['event_id'].n_unique()} events, "
          f"{signals['city'].n_unique()} cities")

    t0 = time.time()

    # Run all sections
    trades, metrics = section_per_trade_pnl(signals, args.out_dir, rng)
    baseline_results = section_baselines(signals, args.out_dir, rng)
    section_stratification(signals, args.events_dir, args.out_dir, rng)
    section_capacity(signals, args.out_dir, rng)
    section_adverse_selection(signals, args.events_dir, args.out_dir, rng)
    section_verdict(metrics, baseline_results)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
