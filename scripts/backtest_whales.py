"""Walk-forward backtest: simulate following top weather traders."""

import argparse
import time

from src.whales.backtest import WhaleBacktester


def _print_report(summary) -> None:
    """Print the full backtest report to stdout."""
    print(f"\n{'='*70}")
    print("  WHALE-TAIL BACKTEST REPORT")
    print(f"{'='*70}")
    print(f"  Train: {summary.train_start} → {summary.train_end}")
    print(f"         {summary.n_train_markets} resolved weather markets")
    print(f"  Test:  {summary.test_start} → {summary.test_end}")
    print(f"         {summary.n_test_markets} resolved weather markets")

    print(f"\n{'='*70}")
    print("  TOP WHALES (identified from training period)")
    print(f"{'='*70}")
    header = f"  {'Address':<18} {'Markets':>7} {'PnL':>14} {'ROI':>8}"
    print(header)
    print("  " + "-" * 50)
    for w in summary.train_whales_top:
        print(
            f"  {w.address[:16]}..  "
            f"{w.n_markets:>5}  "
            f"${w.total_pnl:>12,.0f}  "
            f"{w.roi:>7.1%}"
        )

    print(f"\n{'='*70}")
    print("  BOTTOM WHALES (identified from training period)")
    print(f"{'='*70}")
    print(header)
    print("  " + "-" * 50)
    for w in summary.train_whales_bottom:
        print(
            f"  {w.address[:16]}..  "
            f"{w.n_markets:>5}  "
            f"${w.total_pnl:>12,.0f}  "
            f"{w.roi:>7.1%}"
        )

    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON (test period)")
    print(f"{'='*70}")
    header = (
        f"  {'Strategy':<18} {'Mkts':>5} {'Pos':>5} "
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

    # Best strategy
    if summary.strategies:
        best = max(summary.strategies, key=lambda s: s.sharpe)
        print(f"\n  Best Sharpe: {best.name} ({best.sharpe:.3f})")
        best_pnl = max(summary.strategies, key=lambda s: s.total_pnl)
        print(f"  Best PnL:    {best_pnl.name} (${best_pnl.total_pnl:+,.0f})")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest: simulate following top weather traders."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top whales to identify in training (default: 10).",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=5,
        help="Number of bottom whales to identify in training (default: 5).",
    )
    parser.add_argument(
        "--bet-size",
        type=float,
        default=100.0,
        help="Dollar amount per simulated bet (default: 100).",
    )
    args = parser.parse_args()

    print("Whale-tail backtest")
    print(f"  Top whales:  {args.top_n}")
    print(f"  Bottom whales: {args.bottom_n}")
    print(f"  Bet size:    ${args.bet_size:.0f}")
    print()

    backtester = WhaleBacktester.create(bet_size=args.bet_size)

    t0 = time.monotonic()
    summary = backtester.run(
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        bet_size=args.bet_size,
    )
    elapsed = time.monotonic() - t0
    print(f"\nBacktest completed in {elapsed:.1f}s")

    _print_report(summary)


if __name__ == "__main__":
    main()
