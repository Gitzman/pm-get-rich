"""Walk-forward backtest: simulate following top weather traders.

Uses rolling windows to avoid survivorship bias.  Wallets are re-ranked
each period using only lookback data.
"""

import argparse
import time

from src.whales.backtest import WhaleBacktester, WhaleBacktestSummary


def _print_report(summary: WhaleBacktestSummary) -> None:
    """Print the full per-period and aggregate backtest report."""
    print(f"\n{'='*74}")
    print("  WHALE-TAIL WALK-FORWARD BACKTEST")
    print(f"{'='*74}")
    print(
        f"  Rolling windows: {summary.n_periods} periods  "
        f"({summary.window_lookback_days}d lookback → "
        f"{summary.window_forward_days}d forward)"
    )
    print(f"  Total weather markets: {summary.total_markets}")
    print(f"  Whale persistence (Spearman ρ): {summary.whale_persistence:.3f}")

    # Per-period breakdown for tail-top-5
    tail5 = next(
        (s for s in summary.strategies if s.name == "tail-top-5"), None
    )
    if tail5 and tail5.periods:
        print(f"\n{'='*74}")
        print("  PER-PERIOD RESULTS: tail-top-5")
        print(f"{'='*74}")
        header = (
            f"  {'Period':>3} {'Forward Window':<27} "
            f"{'Mkts':>5} {'Bets':>5} {'PnL':>11} {'Win%':>6}"
        )
        print(header)
        print("  " + "-" * 68)
        for pr in tail5.periods:
            fs = pr.forward_start[:10]
            fe = pr.forward_end[:10]
            pnl_str = f"${pr.total_pnl:+,.0f}"
            print(
                f"  {pr.period_idx:>3}  {fs} → {fe}  "
                f"{pr.n_forward_markets:>5} {pr.n_bets:>5} "
                f"{pnl_str:>11} {pr.win_rate:>5.1%}"
            )

    # Aggregate strategy comparison
    print(f"\n{'='*74}")
    print("  AGGREGATE STRATEGY COMPARISON")
    print(f"{'='*74}")
    header = (
        f"  {'Strategy':<18} {'Mkts':>5} {'Bets':>5} "
        f"{'PnL':>12} {'Win%':>6} {'ROI':>7} "
        f"{'Sharpe':>7} {'MaxDD':>10} {'Edge':>6}"
    )
    print(header)
    print("  " + "-" * 80)
    for s in summary.strategies:
        pnl_str = f"${s.total_pnl:+,.0f}"
        dd_str = f"${s.max_drawdown:,.0f}"
        print(
            f"  {s.name:<18} {s.n_markets:>5} {len(s.positions):>5} "
            f"{pnl_str:>12} {s.win_rate:>5.1%} {s.roi:>6.1%} "
            f"{s.sharpe:>7.3f} {dd_str:>10} {s.avg_edge:>5.3f}"
        )

    if summary.strategies:
        best = max(summary.strategies, key=lambda s: s.sharpe)
        print(f"\n  Best Sharpe: {best.name} ({best.sharpe:.3f})")
        best_pnl = max(summary.strategies, key=lambda s: s.total_pnl)
        print(f"  Best PnL:    {best_pnl.name} (${best_pnl.total_pnl:+,.0f})")

    print(f"\n  Whale persistence: {summary.whale_persistence:.3f}")
    if summary.whale_persistence > 0.3:
        print("  → Winners tend to persist across periods (positive signal)")
    elif summary.whale_persistence < -0.1:
        print("  → Winner rankings invert between periods (mean-reversion)")
    else:
        print("  → Weak persistence — rankings shuffle between periods")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward whale-tailing backtest with rolling windows."
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Lookback window in days for wallet ranking (default: 90).",
    )
    parser.add_argument(
        "--forward-days",
        type=int,
        default=30,
        help="Forward window in days for simulation (default: 30).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top whales to track per window (default: 10).",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=5,
        help="Number of bottom whales to fade per window (default: 5).",
    )
    parser.add_argument(
        "--bet-size",
        type=float,
        default=100.0,
        help="Dollar amount per simulated bet (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for baseline strategy (default: 42).",
    )
    args = parser.parse_args()

    print("Whale-tail walk-forward backtest (rolling windows)")
    print(f"  Lookback:   {args.lookback_days} days")
    print(f"  Forward:    {args.forward_days} days")
    print(f"  Top whales: {args.top_n}")
    print(f"  Bottom:     {args.bottom_n}")
    print(f"  Bet size:   ${args.bet_size:.0f}")
    print()

    backtester = WhaleBacktester.create(bet_size=args.bet_size)

    t0 = time.monotonic()
    summary = backtester.run(
        lookback_days=args.lookback_days,
        forward_days=args.forward_days,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        bet_size=args.bet_size,
        random_seed=args.seed,
    )
    elapsed = time.monotonic() - t0
    print(f"\nBacktest completed in {elapsed:.1f}s")

    _print_report(summary)


if __name__ == "__main__":
    main()
