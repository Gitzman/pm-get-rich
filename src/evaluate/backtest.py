"""Backtest runner for evaluating Chronos forecasts against resolved markets."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from src.evaluate.metrics import (
    CalibrationBin,
    ProfitResult,
    brier_score,
    calibration_curve,
    calibration_error,
    mean_brier_score,
    profit_simulation,
    sharpe_like_ratio,
)
from src.forecast.chronos import ChronosForecaster
from src.store.db import connect, get_price_series


@dataclass
class MarketResult:
    market_id: str
    question: str
    outcome: float
    forecast_median: float
    forecast_p10: float
    forecast_p90: float
    last_price: float
    brier: float
    n_datapoints: int


@dataclass
class BacktestSummary:
    results: list[MarketResult]
    avg_brier: float
    calibration_bins: list[CalibrationBin]
    cal_error: float
    profit: ProfitResult
    sharpe: float


class BacktestRunner:
    """Run backtests against historical resolved markets."""

    def __init__(
        self,
        forecaster: ChronosForecaster,
        con: duckdb.DuckDBPyConnection,
        min_datapoints: int = 50,
    ) -> None:
        self._forecaster = forecaster
        self._con = con
        self._min_datapoints = min_datapoints

    @classmethod
    def create(
        cls,
        model_name: str | None = None,
        device: str | None = None,
        min_datapoints: int = 50,
    ) -> BacktestRunner:
        """Create a BacktestRunner with default DB and model."""
        forecaster = ChronosForecaster.from_pretrained(model_name, device)
        con = connect()
        return cls(forecaster, con, min_datapoints)

    def select_markets(self, max_markets: int = 1000) -> list[dict]:
        """Select resolved markets with sufficient price history."""
        rows = self._con.execute(
            """
            SELECT m.*, t.datapoints
            FROM markets m
            JOIN (
                SELECT market_id, count(*) as datapoints
                FROM trades
                GROUP BY market_id
                HAVING count(*) >= ?
            ) t ON m.id = t.market_id
            WHERE m.closed = true AND m.outcome IS NOT NULL
            ORDER BY t.datapoints DESC
            LIMIT ?
            """,
            [self._min_datapoints, max_markets],
        ).fetchall()
        columns = [desc[0] for desc in self._con.description]
        return [dict(zip(columns, row)) for row in rows]

    def run(
        self,
        max_markets: int = 1000,
        divergence_threshold: float = 0.10,
        bet_size: float = 100.0,
        prediction_length: int = 64,
    ) -> BacktestSummary:
        """Run the full backtest pipeline.

        For each selected market:
        1. Get price history
        2. Use Chronos to forecast from the history
        3. Compare final forecast value to actual outcome
        4. Compute Brier score

        Then aggregate into calibration and profit metrics.
        """
        markets = self.select_markets(max_markets)
        if not markets:
            raise RuntimeError("No resolved markets with sufficient data found")

        print(f"Selected {len(markets)} resolved markets for backtesting")

        results: list[MarketResult] = []
        for i, market in enumerate(markets):
            market_id = market["id"]
            question = market.get("question", market_id)
            outcome = float(market["outcome"])

            series = get_price_series(self._con, market_id)
            if len(series) < self._min_datapoints:
                continue

            df = pd.DataFrame(series)
            last_price = df["price"].iloc[-1]

            try:
                forecast_df = self._forecaster.predict_market(
                    df[["timestamp", "price"]].assign(market_id=market_id),
                    prediction_length=prediction_length,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(markets)}] {market_id}: forecast failed ({e})")
                continue

            # Extract final forecast step quantiles
            # Chronos predict_df returns columns like 0.1, 0.5, 0.9
            final_row = forecast_df.iloc[-1]
            forecast_median = float(final_row.get(0.5, final_row.get("0.5", 0.5)))
            forecast_p10 = float(final_row.get(0.1, final_row.get("0.1", 0.1)))
            forecast_p90 = float(final_row.get(0.9, final_row.get("0.9", 0.9)))

            # Clamp to [0, 1] since these are probabilities
            forecast_median = max(0.0, min(1.0, forecast_median))
            forecast_p10 = max(0.0, min(1.0, forecast_p10))
            forecast_p90 = max(0.0, min(1.0, forecast_p90))

            brier = brier_score(forecast_median, outcome)

            results.append(
                MarketResult(
                    market_id=market_id,
                    question=question[:80],
                    outcome=outcome,
                    forecast_median=forecast_median,
                    forecast_p10=forecast_p10,
                    forecast_p90=forecast_p90,
                    last_price=last_price,
                    brier=brier,
                    n_datapoints=len(series),
                )
            )

            if (i + 1) % 50 == 0 or i + 1 == len(markets):
                print(f"  [{i+1}/{len(markets)}] processed")

        if not results:
            raise RuntimeError("No markets produced valid forecasts")

        # Aggregate metrics
        forecasts = [r.forecast_median for r in results]
        outcomes = [r.outcome for r in results]
        prices = [r.last_price for r in results]
        brier_scores = [r.brier for r in results]

        avg_brier = mean_brier_score(brier_scores)
        cal_bins = calibration_curve(forecasts, outcomes)
        cal_err = calibration_error(cal_bins)
        profit = profit_simulation(
            forecasts, outcomes, prices,
            bet_size=bet_size,
            divergence_threshold=divergence_threshold,
        )

        # Compute per-bet PnL for Sharpe
        pnl_per_bet: list[float] = []
        for f, o, p in zip(forecasts, outcomes, prices):
            div = f - p
            if abs(div) < divergence_threshold:
                continue
            if div > 0:
                pnl_per_bet.append(
                    bet_size * (1.0 / p - 1.0) if o == 1 else -bet_size
                )
            else:
                pnl_per_bet.append(
                    bet_size * (1.0 / (1.0 - p) - 1.0) if o == 0 else -bet_size
                )

        sharpe = sharpe_like_ratio(pnl_per_bet)

        # Store results in DuckDB
        self._store_results(results)

        return BacktestSummary(
            results=results,
            avg_brier=avg_brier,
            calibration_bins=cal_bins,
            cal_error=cal_err,
            profit=profit,
            sharpe=sharpe,
        )

    def _store_results(self, results: list[MarketResult]) -> None:
        """Store backtest results in a DuckDB table."""
        self._con.execute("DROP TABLE IF EXISTS backtest_results")
        self._con.execute("""
            CREATE TABLE backtest_results (
                market_id VARCHAR,
                forecast_median DOUBLE,
                forecast_p10 DOUBLE,
                forecast_p90 DOUBLE,
                actual_outcome DOUBLE,
                brier_score DOUBLE
            )
        """)
        for r in results:
            self._con.execute(
                """
                INSERT INTO backtest_results VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    r.market_id,
                    r.forecast_median,
                    r.forecast_p10,
                    r.forecast_p90,
                    r.outcome,
                    r.brier,
                ],
            )
        print(f"  Stored {len(results)} results in backtest_results table")
