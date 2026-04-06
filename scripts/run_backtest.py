"""Run backtesting against historical Polymarket data."""

import argparse
import time

from src.config import settings
from src.evaluate.backtest import BacktestRunner
from src.evaluate.report import print_full_report, save_html_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Chronos forecasts against resolved Polymarket markets."
    )
    parser.add_argument(
        "--markets",
        type=int,
        default=1000,
        help="Maximum number of resolved markets to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--min-datapoints",
        type=int,
        default=50,
        help="Minimum price history datapoints per market (default: 50).",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=0.10,
        help="Minimum |forecast - price| to trigger a simulated bet (default: 0.10).",
    )
    parser.add_argument(
        "--bet-size",
        type=float,
        default=100.0,
        help="Dollar amount per simulated bet (default: 100).",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=64,
        help="Number of future steps for Chronos to predict (default: 64).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Chronos model name (default: {settings.chronos_model}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Device for inference (default: {settings.device}).",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Save HTML report to this path (e.g., data/reports/backtest.html).",
    )
    args = parser.parse_args()

    print(f"Model:               {args.model or settings.chronos_model}")
    print(f"Data source:         {settings.duckdb_path}")
    print(f"Max markets:         {args.markets}")
    print(f"Min datapoints:      {args.min_datapoints}")
    print(f"Divergence threshold:{args.divergence_threshold}")
    print(f"Bet size:            ${args.bet_size:.0f}")
    print()

    runner = BacktestRunner.create(
        model_name=args.model,
        device=args.device,
        min_datapoints=args.min_datapoints,
    )

    t0 = time.monotonic()
    summary = runner.run(
        max_markets=args.markets,
        divergence_threshold=args.divergence_threshold,
        bet_size=args.bet_size,
        prediction_length=args.prediction_length,
    )
    elapsed = time.monotonic() - t0
    print(f"\nBacktest completed in {elapsed:.1f}s")

    print_full_report(summary)

    if args.html:
        from pathlib import Path

        save_html_report(summary, Path(args.html))


if __name__ == "__main__":
    main()
