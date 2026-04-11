# MIT License — pmgetrich project
# Backtest runner kept separate from LGPL NautilusTrader components.

"""
TPP model vs volume baseline on Polymarket weather temperature markets.

Runs both strategies on the same set of weather markets using NautilusTrader's
realistic fill simulation (native trade tick data from Polymarket API).
Compares results against our statistical backtest findings.
"""

# ruff: noqa: E402

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

if __package__ in {None, ""}:
    from _script_helpers import ensure_repo_root
else:
    from ._script_helpers import ensure_repo_root

ensure_repo_root(__file__)

from prediction_market_extensions.backtesting._experiments import build_replay_experiment
from prediction_market_extensions.backtesting._experiments import run_experiment
from prediction_market_extensions.backtesting._prediction_market_backtest import MarketReportConfig
from prediction_market_extensions.backtesting._prediction_market_runner import MarketDataConfig
from prediction_market_extensions.backtesting._replay_specs import TradeReplay
from prediction_market_extensions.backtesting._timing_harness import timing_harness
from prediction_market_extensions.backtesting.data_sources import Native, Polymarket, TradeTick

# --- Paths to model artifacts (relative to pmgetrich repo root) ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(REPO_ROOT / "data" / "models" / "tpp.onnx")
VOCAB_PATH = str(REPO_ROOT / "data" / "reports" / "generalization" / "cross_market" / "vocab.json")

NAME = "polymarket_weather_tpp_vs_volume"

DESCRIPTION = (
    "TPP ONNX model vs volume baseline on Polymarket weather markets "
    "(BUY-only, 5 cities, native trade ticks)"
)

EMIT_HTML = True
CHART_OUTPUT_PATH = "output"
DETAIL_PLOT_PANELS = (
    "total_equity",
    "equity",
    "market_pnl",
    "periodic_pnl",
    "yes_price",
    "allocation",
    "total_drawdown",
    "drawdown",
    "total_rolling_sharpe",
    "rolling_sharpe",
    "total_cash_equity",
    "cash_equity",
    "monthly_returns",
    "total_brier_advantage",
    "brier_advantage",
)
SUMMARY_PLOT_PANELS = (
    "total_equity",
    "total_drawdown",
    "total_rolling_sharpe",
    "total_cash_equity",
    "total_brier_advantage",
    "periodic_pnl",
    "monthly_returns",
)

DATA = MarketDataConfig(
    platform=Polymarket,
    data_type=TradeTick,
    vendor=Native,
    sources=(
        "gamma:https://gamma-api.polymarket.com",
        "trades:https://data-api.polymarket.com",
        "clob:https://clob.polymarket.com",
    ),
)

# Weather temperature markets from our holdout set (5 cities, diverse dates)
# Each market has ~11 buckets; we use the first bucket (token_index=0) as the
# instrument, matching our backtest setup.
REPLAYS = (
    # NYC markets
    TradeReplay(
        market_slug="highest-temperature-in-nyc-on-march-24-2026-37forbelow",
        lookback_days=3,
        end_time="2026-03-24T12:00:00Z",
        metadata={"city": "NYC", "event_id": "289197"},
    ),
    TradeReplay(
        market_slug="highest-temperature-in-nyc-on-march-27-2026-49forbelow",
        lookback_days=3,
        end_time="2026-03-27T12:00:00Z",
        metadata={"city": "NYC", "event_id": "299417"},
    ),
    # London
    TradeReplay(
        market_slug="highest-temperature-in-london-on-march-24-2026-9corbelow",
        lookback_days=3,
        end_time="2026-03-24T12:00:00Z",
        metadata={"city": "London", "event_id": "289183"},
    ),
    # Chicago
    TradeReplay(
        market_slug="highest-temperature-in-chicago-on-march-24-2026-37forbelow",
        lookback_days=3,
        end_time="2026-03-24T12:00:00Z",
        metadata={"city": "Chicago", "event_id": "289201"},
    ),
    # NYC (different date for walk-forward)
    TradeReplay(
        market_slug="highest-temperature-in-nyc-on-march-25-2026-41forbelow",
        lookback_days=3,
        end_time="2026-03-25T12:00:00Z",
        metadata={"city": "NYC", "event_id": "292566"},
    ),
)


# --- TPP Strategy ---
TPP_STRATEGY_CONFIGS = [
    {
        "strategy_path": "strategies:TradeTickTPPSignalStrategy",
        "config_path": "strategies:TradeTickTPPSignalConfig",
        "config": {
            "trade_size": Decimal("100"),
            "model_path": MODEL_PATH,
            "vocab_path": VOCAB_PATH,
            "city_name": "NYC",
            "n_buckets": 11,
            "hours_to_resolution": 72.0,
            "context_length": 128,
            "confidence_threshold": 0.55,
            "cooldown_ticks": 50,
            "take_profit": 0.015,
            "stop_loss": 0.02,
        },
    }
]

# --- Volume Baseline Strategy ---
VOLUME_STRATEGY_CONFIGS = [
    {
        "strategy_path": "strategies:TradeTickVolumeBaselineStrategy",
        "config_path": "strategies:TradeTickVolumeBaselineConfig",
        "config": {
            "trade_size": Decimal("100"),
            "volume_window": 60,
            "volume_percentile": 90.0,
            "cooldown_ticks": 50,
            "take_profit": 0.015,
            "stop_loss": 0.02,
        },
    }
]

REPORT = MarketReportConfig(
    count_key="trades",
    count_label="Trades",
    pnl_label="PnL (USDC)",
    summary_report=True,
    summary_report_path=f"output/{NAME}_summary.html",
    summary_plot_panels=SUMMARY_PLOT_PANELS,
)

EMPTY_MESSAGE = "No weather market replays met the trade-tick requirements."
PARTIAL_MESSAGE = "Completed {completed} of {total} weather market replays."


def build_tpp_experiment():
    return build_replay_experiment(
        name=f"{NAME}_tpp",
        description=f"{DESCRIPTION} — TPP model",
        data=DATA,
        replays=REPLAYS,
        strategy_configs=TPP_STRATEGY_CONFIGS,
        initial_cash=100.0,
        probability_window=80,
        min_trades=10,
        min_price_range=0.005,
        report=REPORT,
        empty_message=EMPTY_MESSAGE,
        partial_message=PARTIAL_MESSAGE,
        emit_html=EMIT_HTML,
        chart_output_path=CHART_OUTPUT_PATH,
        detail_plot_panels=DETAIL_PLOT_PANELS,
        return_summary_series=True,
        multi_replay_mode="joint_portfolio",
    )


def build_volume_experiment():
    return build_replay_experiment(
        name=f"{NAME}_volume",
        description=f"{DESCRIPTION} — Volume baseline",
        data=DATA,
        replays=REPLAYS,
        strategy_configs=VOLUME_STRATEGY_CONFIGS,
        initial_cash=100.0,
        probability_window=80,
        min_trades=10,
        min_price_range=0.005,
        report=MarketReportConfig(
            count_key="trades",
            count_label="Trades",
            pnl_label="PnL (USDC)",
            summary_report=True,
            summary_report_path=f"output/{NAME}_volume_summary.html",
            summary_plot_panels=SUMMARY_PLOT_PANELS,
        ),
        empty_message=EMPTY_MESSAGE,
        partial_message=PARTIAL_MESSAGE,
        emit_html=EMIT_HTML,
        chart_output_path=CHART_OUTPUT_PATH,
        detail_plot_panels=DETAIL_PLOT_PANELS,
        return_summary_series=True,
        multi_replay_mode="joint_portfolio",
    )


# Default experiment (TPP) — used by main.py menu
EXPERIMENT = build_tpp_experiment()


@timing_harness
def run() -> None:
    """Run both TPP and volume backtests for comparison."""
    print("=" * 70)
    print("Running TPP model backtest...")
    print("=" * 70)
    tpp_exp = build_tpp_experiment()
    run_experiment(tpp_exp)

    print("\n" + "=" * 70)
    print("Running volume baseline backtest...")
    print("=" * 70)
    vol_exp = build_volume_experiment()
    run_experiment(vol_exp)

    print("\n" + "=" * 70)
    print("Both backtests complete. Compare results in output/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    run()
