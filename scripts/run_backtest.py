"""Run backtesting against historical Polymarket data."""

from src.config import settings


def main() -> None:
    print(f"Running backtest with model: {settings.chronos_model}")
    print(f"Data source: {settings.duckdb_path}")
    raise NotImplementedError("run_backtest not yet implemented")


if __name__ == "__main__":
    main()
