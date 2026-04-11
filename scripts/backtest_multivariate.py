"""Backtest comparing multivariate (trade features) vs price-only Chronos forecasts.

Tests whether adding trade-activity features (whale_net_flow, volume_zscore,
smart_dumb_ratio, maker_taker_imbalance) improves Brier scores on resolved
weather markets compared to the baseline price-only model.

Usage:
    uv run python scripts/backtest_multivariate.py [--markets N] [--device cpu]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import settings
from src.evaluate.metrics import brier_score, mean_brier_score
from src.forecast.chronos import ChronosForecaster
from src.forecast.features import prepare_multivariate_series, prepare_series
from src.store.db import connect, get_price_series
from src.whales.features import get_whale_addresses

COVARIATE_COLS = [
    "volume_zscore",
    "maker_taker_imbalance",
    "whale_net_flow",
    "smart_dumb_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multivariate vs price-only Chronos forecasts."
    )
    parser.add_argument(
        "--markets", type=int, default=100,
        help="Max resolved markets to evaluate (default: 100).",
    )
    parser.add_argument(
        "--min-datapoints", type=int, default=50,
        help="Minimum price history datapoints per market (default: 50).",
    )
    parser.add_argument(
        "--prediction-length", type=int, default=64,
        help="Chronos prediction length (default: 64).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Chronos model (default: {settings.chronos_model}).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help=f"Device (default: {settings.device}).",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top whale wallets (default: 20).",
    )
    args = parser.parse_args()

    con = connect()

    # Compute whale rankings once
    print("Computing whale rankings...")
    t0 = time.monotonic()
    top_whales, bottom_whales = get_whale_addresses(
        con, top_n=args.top_n, bottom_n=args.top_n
    )
    print(f"  Found {len(top_whales)} top, {len(bottom_whales)} bottom whales "
          f"in {time.monotonic() - t0:.1f}s")

    # Select resolved markets with sufficient history
    markets = con.execute(
        """
        SELECT m.id, m.question,
               ROUND(CAST(
                   json_extract_string(m.outcome_prices, '$[0]') AS DOUBLE
               )) AS outcome,
               t.datapoints
        FROM markets m
        JOIN (
            SELECT market_id, count(*) AS datapoints
            FROM trades
            GROUP BY market_id
            HAVING count(*) >= ?
        ) t ON m.id = t.market_id
        WHERE m.closed = 1
          AND m.outcome_prices IS NOT NULL
          AND m.outcome_prices != ''
        ORDER BY t.datapoints DESC
        LIMIT ?
        """,
        [args.min_datapoints, args.markets],
    ).fetchall()

    if not markets:
        print("No resolved markets with sufficient data found.")
        con.close()
        return

    print(f"Selected {len(markets)} resolved markets\n")

    # Load model
    print("Loading Chronos model...")
    forecaster = ChronosForecaster.from_pretrained(
        model_name=args.model, device_map=args.device
    )

    # Run comparison
    results = []
    for i, (market_id, question, outcome, n_pts) in enumerate(markets):
        outcome = float(outcome)

        # --- Price-only forecast ---
        raw = get_price_series(con, market_id)
        if len(raw) < args.min_datapoints:
            continue

        price_df = prepare_series(
            pd.DataFrame(raw), freq="1h",
            timestamp_col="timestamp", price_col="price",
            market_id=market_id,
        )

        try:
            forecast_price = forecaster.predict_market(
                price_df, prediction_length=args.prediction_length,
            )
        except Exception as e:
            print(f"  [{i+1}] {market_id}: price-only failed ({e})")
            continue

        final_price = forecast_price.iloc[-1]
        median_price = float(
            final_price.get(0.5, final_price.get("0.5", 0.5))
        )
        median_price = max(0.0, min(1.0, median_price))
        brier_price = brier_score(median_price, outcome)

        # --- Multivariate forecast ---
        mv_df = prepare_multivariate_series(
            con, market_id, freq="1h",
            top_whales=top_whales, bottom_whales=bottom_whales,
        )

        if mv_df.empty or len(mv_df) < 10:
            print(f"  [{i+1}] {market_id}: multivariate series too short")
            continue

        try:
            forecast_mv = forecaster.predict_market(
                mv_df, prediction_length=args.prediction_length,
                covariates=COVARIATE_COLS,
            )
        except Exception as e:
            print(f"  [{i+1}] {market_id}: multivariate failed ({e})")
            continue

        final_mv = forecast_mv.iloc[-1]
        median_mv = float(final_mv.get(0.5, final_mv.get("0.5", 0.5)))
        median_mv = max(0.0, min(1.0, median_mv))
        brier_mv = brier_score(median_mv, outcome)

        results.append({
            "market_id": market_id,
            "question": (question or market_id)[:60],
            "outcome": outcome,
            "brier_price_only": brier_price,
            "brier_multivariate": brier_mv,
            "improvement": brier_price - brier_mv,
        })

        if (i + 1) % 25 == 0 or i + 1 == len(markets):
            print(f"  [{i+1}/{len(markets)}] processed")

    con.close()

    if not results:
        print("No valid results produced.")
        return

    # Report
    df = pd.DataFrame(results)
    avg_price = mean_brier_score(df["brier_price_only"].tolist())
    avg_mv = mean_brier_score(df["brier_multivariate"].tolist())
    n_improved = (df["improvement"] > 0).sum()

    print("\n" + "=" * 60)
    print("MULTIVARIATE vs PRICE-ONLY COMPARISON")
    print("=" * 60)
    print(f"Markets evaluated:     {len(df)}")
    print(f"Avg Brier (price-only):{avg_price:.4f}")
    print(f"Avg Brier (multivar):  {avg_mv:.4f}")
    print(f"Improvement:           {avg_price - avg_mv:+.4f}")
    print(f"Markets improved:      {n_improved}/{len(df)} "
          f"({100 * n_improved / len(df):.0f}%)")
    print()

    # Top improvements
    top = df.nlargest(5, "improvement")
    print("Top 5 improvements:")
    for _, row in top.iterrows():
        print(f"  {row['question']}: "
              f"{row['brier_price_only']:.3f} -> {row['brier_multivariate']:.3f} "
              f"({row['improvement']:+.3f})")

    # Top regressions
    bot = df.nsmallest(5, "improvement")
    print("\nTop 5 regressions:")
    for _, row in bot.iterrows():
        print(f"  {row['question']}: "
              f"{row['brier_price_only']:.3f} -> {row['brier_multivariate']:.3f} "
              f"({row['improvement']:+.3f})")


if __name__ == "__main__":
    main()
