from __future__ import annotations

import pandas as pd
import torch
from chronos import Chronos2Pipeline

from src.config import settings

MIN_SERIES_LENGTH = 10


class ChronosForecaster:
    """Wrapper around Chronos 2 for market price forecasting."""

    def __init__(self, pipeline: Chronos2Pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | None = None,
        device_map: str | None = None,
    ) -> ChronosForecaster:
        """Load a pretrained Chronos 2 model.

        Args:
            model_name: HuggingFace model ID. Defaults to config.chronos_model.
            device_map: Device placement ('cuda', 'cpu', 'auto').
                Defaults to config.device.
        """
        model_name = model_name or settings.chronos_model
        device_map = device_map or settings.device
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        return cls(pipeline)

    def predict_market(
        self,
        price_series: pd.DataFrame,
        prediction_length: int = 64,
        quantile_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """Forecast a single market price series.

        Args:
            price_series: DataFrame with columns: timestamp, price.
                Optionally includes 'market_id'.
            prediction_length: Number of future steps to predict.
            quantile_levels: Quantile levels for probabilistic forecasts.

        Returns:
            DataFrame with forecast timestamps and quantile columns.
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        df = _validate_series(price_series)

        if "market_id" not in df.columns:
            df = df.assign(market_id="default")

        return self._pipeline.predict_df(
            df,
            id_column="market_id",
            timestamp_column="timestamp",
            target="price",
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

    def batch_predict(
        self,
        series_list: list[pd.DataFrame],
        prediction_length: int = 64,
        quantile_levels: list[float] | None = None,
    ) -> list[pd.DataFrame]:
        """Forecast multiple market price series in a single batch.

        Args:
            series_list: List of DataFrames, each with columns:
                timestamp, price. Each may include 'market_id'.
            prediction_length: Number of future steps to predict.
            quantile_levels: Quantile levels for probabilistic forecasts.

        Returns:
            List of forecast DataFrames, one per input series.
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]

        combined_parts: list[pd.DataFrame] = []
        id_map: dict[str, int] = {}

        for i, series in enumerate(series_list):
            df = _validate_series(series)
            original_id = (
                df["market_id"].iloc[0]
                if "market_id" in df.columns
                else f"market_{i}"
            )
            unique_id = f"{original_id}__{i}"
            id_map[unique_id] = i
            combined_parts.append(df.assign(market_id=unique_id))

        combined = pd.concat(combined_parts, ignore_index=True)

        forecast_df = self._pipeline.predict_df(
            combined,
            id_column="market_id",
            timestamp_column="timestamp",
            target="price",
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        results: list[pd.DataFrame] = [pd.DataFrame()] * len(series_list)
        for uid, idx in id_map.items():
            mask = forecast_df["market_id"] == uid
            results[idx] = forecast_df.loc[mask].reset_index(drop=True)

        return results


def _validate_series(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a price series DataFrame.

    Checks for required columns, converts timestamps, drops NaN prices,
    enforces minimum length, and sorts by time.
    """
    required = {"timestamp", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    nan_count = df["price"].isna().sum()
    if nan_count > 0:
        df = df.dropna(subset=["price"])

    if len(df) < MIN_SERIES_LENGTH:
        raise ValueError(
            f"Series too short: {len(df)} rows (minimum {MIN_SERIES_LENGTH})"
        )

    return df.sort_values("timestamp").reset_index(drop=True)
