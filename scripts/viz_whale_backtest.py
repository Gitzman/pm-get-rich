"""Visualize whale-tail walk-forward backtest results.

Generates charts from WhaleBacktester output and saves PNGs to data/reports/.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.whales.backtest import WhaleBacktester, WhaleBacktestSummary

REPORT_DIR = Path("data/reports")


def _period_end_dates(summary: WhaleBacktestSummary, strategy_name: str) -> list[str]:
    """Extract forward-end dates for a strategy's periods."""
    strat = next((s for s in summary.strategies if s.name == strategy_name), None)
    if not strat:
        return []
    return [pr.forward_end[:10] for pr in strat.periods]


def chart_cumulative_pnl(summary: WhaleBacktestSummary, out: Path) -> None:
    """Chart 1: Cumulative PnL over time for all 5 strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "tail-top-5": "#2196F3",
        "tail-top-10": "#4CAF50",
        "fade-bottom-5": "#FF9800",
        "combined": "#9C27B0",
        "random": "#9E9E9E",
    }
    linestyles = {
        "tail-top-5": "-",
        "tail-top-10": "-",
        "fade-bottom-5": "--",
        "combined": "-",
        "random": ":",
    }

    for strat in summary.strategies:
        if not strat.periods:
            continue
        dates = [pr.forward_end[:10] for pr in strat.periods]
        cum_pnl = []
        running = 0.0
        for pr in strat.periods:
            running += pr.total_pnl
            cum_pnl.append(running)

        ax.plot(
            range(len(dates)),
            cum_pnl,
            label=f"{strat.name} (${strat.total_pnl:+,.0f})",
            color=colors.get(strat.name, "#333"),
            linestyle=linestyles.get(strat.name, "-"),
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Period End Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Whale-Tail Backtest: Cumulative PnL by Strategy")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Use dates from first strategy with periods for x-axis labels
    for strat in summary.strategies:
        if strat.periods:
            dates = [pr.forward_end[:10] for pr in strat.periods]
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
            break

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def chart_period_pnl_bars(summary: WhaleBacktestSummary, out: Path) -> None:
    """Chart 2: Per-period PnL bars for tail-top-5."""
    strat = next((s for s in summary.strategies if s.name == "tail-top-5"), None)
    if not strat or not strat.periods:
        print("  Skipped: no tail-top-5 periods")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    dates = [pr.forward_end[:10] for pr in strat.periods]
    pnls = [pr.total_pnl for pr in strat.periods]
    bar_colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]

    ax.bar(range(len(dates)), pnls, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Period End Date")
    ax.set_ylabel("PnL ($)")
    ax.set_title("tail-top-5: Per-Period PnL")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def chart_win_rate(summary: WhaleBacktestSummary, out: Path) -> None:
    """Chart 3: Per-period win rate for tail-top-5."""
    strat = next((s for s in summary.strategies if s.name == "tail-top-5"), None)
    if not strat or not strat.periods:
        print("  Skipped: no tail-top-5 periods")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    dates = [pr.forward_end[:10] for pr in strat.periods]
    win_rates = [pr.win_rate * 100 for pr in strat.periods]

    ax.plot(
        range(len(dates)),
        win_rates,
        color="#2196F3",
        linewidth=2,
        marker="o",
        markersize=5,
    )
    ax.axhline(y=50, color="gray", linewidth=1, linestyle="--", label="50% baseline")
    ax.set_xlabel("Period End Date")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("tail-top-5: Per-Period Win Rate")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for i, wr in enumerate(win_rates):
        ax.annotate(
            f"{wr:.0f}%",
            (i, wr),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def chart_whale_persistence(summary: WhaleBacktestSummary, out: Path) -> None:
    """Chart 4: Whale persistence heatmap -- top-10 address overlap between adjacent periods."""
    # Collect top-10 address lists per period from tail-top-10 strategy
    strat = next((s for s in summary.strategies if s.name == "tail-top-10"), None)
    if not strat or len(strat.periods) < 2:
        print("  Skipped: need >= 2 periods for persistence heatmap")
        return

    # Use top_whale_addrs from periods -- but tail-top-10 stores top-5 in
    # top_whale_addrs. Reconstruct from positions instead: unique whale addresses
    # per period.
    period_addrs: list[set[str]] = []
    for pr in strat.periods:
        addrs = {p.whale_address for p in pr.positions}
        period_addrs.append(addrs)

    n = len(period_addrs)
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if period_addrs[i] and period_addrs[j]:
                overlap[i, j] = len(period_addrs[i] & period_addrs[j])
            else:
                overlap[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(overlap, cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="# Overlapping Whale Addresses")

    dates = [pr.forward_end[:10] for pr in strat.periods]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(dates, fontsize=8)
    ax.set_title(
        f"Whale Persistence Heatmap (avg Spearman rho={summary.whale_persistence:.3f})"
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Period")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = int(overlap[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                    color="white" if val > overlap.max() * 0.6 else "black")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def chart_strategy_comparison(summary: WhaleBacktestSummary, out: Path) -> None:
    """Chart 5: Strategy comparison -- grouped bar chart of total PnL, Sharpe, win rate, ROI."""
    if not summary.strategies:
        print("  Skipped: no strategies")
        return

    names = [s.name for s in summary.strategies]
    n = len(names)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#9E9E9E"]
    bar_colors = colors[:n]

    # Total PnL
    ax = axes[0, 0]
    pnls = [s.total_pnl for s in summary.strategies]
    bar_c = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
    ax.bar(x, pnls, color=bar_c, edgecolor="white")
    ax.set_title("Total PnL ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(pnls):
        ax.text(i, v, f"${v:+,.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    # Sharpe
    ax = axes[0, 1]
    sharpes = [s.sharpe for s in summary.strategies]
    ax.bar(x, sharpes, color=bar_colors, edgecolor="white")
    ax.set_title("Sharpe Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(sharpes):
        ax.text(i, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    # Win Rate
    ax = axes[1, 0]
    wrs = [s.win_rate * 100 for s in summary.strategies]
    ax.bar(x, wrs, color=bar_colors, edgecolor="white")
    ax.set_title("Win Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(y=50, color="gray", linewidth=1, linestyle="--")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(wrs):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # ROI
    ax = axes[1, 1]
    rois = [s.roi * 100 for s in summary.strategies]
    bar_c_roi = ["#4CAF50" if r >= 0 else "#F44336" for r in rois]
    ax.bar(x, rois, color=bar_c_roi, edgecolor="white")
    ax.set_title("ROI (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(rois):
        ax.text(i, v, f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)

    fig.suptitle("Whale-Tail Backtest: Strategy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize whale-tail backtest results."
    )
    parser.add_argument(
        "--lookback-days", type=int, default=90,
        help="Lookback window days (default: 90).",
    )
    parser.add_argument(
        "--forward-days", type=int, default=30,
        help="Forward window days (default: 30).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Top whales to track (default: 10).",
    )
    parser.add_argument(
        "--bottom-n", type=int, default=5,
        help="Bottom whales to fade (default: 5).",
    )
    parser.add_argument(
        "--bet-size", type=float, default=100.0,
        help="Bet size in dollars (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running whale backtest...")
    t0 = time.monotonic()
    backtester = WhaleBacktester.create(bet_size=args.bet_size)
    summary = backtester.run(
        lookback_days=args.lookback_days,
        forward_days=args.forward_days,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        bet_size=args.bet_size,
        random_seed=args.seed,
    )
    elapsed = time.monotonic() - t0
    print(f"Backtest completed in {elapsed:.1f}s\n")

    print("Generating charts...")
    chart_cumulative_pnl(summary, REPORT_DIR / "whale_cumulative_pnl.png")
    chart_period_pnl_bars(summary, REPORT_DIR / "whale_period_pnl.png")
    chart_win_rate(summary, REPORT_DIR / "whale_win_rate.png")
    chart_whale_persistence(summary, REPORT_DIR / "whale_persistence_heatmap.png")
    chart_strategy_comparison(summary, REPORT_DIR / "whale_strategy_comparison.png")
    print("\nAll charts saved to data/reports/")


if __name__ == "__main__":
    main()
