from __future__ import annotations

import pandas as pd


def resample_uniform(
    df: pd.DataFrame,
    freq: str = "1h",
    timestamp_col: str = "timestamp",
    price_col: str = "price",
) -> pd.DataFrame:
    """Resample a price series to uniform time intervals.

    Uses last-value aggregation and forward-fills gaps.

    Args:
        df: DataFrame with timestamp and price columns.
        freq: Target frequency ('1h' for hourly, '1D' for daily).
        timestamp_col: Name of the timestamp column.
        price_col: Name of the price column.

    Returns:
        DataFrame resampled to uniform intervals.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df = df.set_index(timestamp_col).sort_index()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    resampled = df[numeric_cols].resample(freq).last().ffill().bfill()

    for col in df.columns:
        if col not in numeric_cols:
            resampled[col] = df[col].resample(freq).last().ffill().bfill()

    return resampled.reset_index()


def validate_prices(
    df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    """Validate YES-token prices are in [0, 1] range.

    Clamps out-of-range values rather than dropping rows.
    """
    df = df.copy()
    df[price_col] = df[price_col].clip(lower=0.0, upper=1.0)
    return df


def prepare_series(
    df: pd.DataFrame,
    freq: str = "1h",
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    market_id: str | None = None,
    volume_col: str | None = None,
) -> pd.DataFrame:
    """Full preparation pipeline for Chronos input.

    Resamples to uniform intervals, validates price range, and formats
    with standard column names expected by ChronosForecaster.

    Args:
        df: Raw price series DataFrame.
        freq: Target resampling frequency.
        timestamp_col: Name of the timestamp column.
        price_col: Name of the price column.
        market_id: Optional market identifier to add.
        volume_col: Optional volume column to include as covariate.

    Returns:
        DataFrame with columns: timestamp, price, and optionally
        market_id and volume.
    """
    prepared = resample_uniform(
        df, freq=freq, timestamp_col=timestamp_col, price_col=price_col
    )
    prepared = validate_prices(prepared, price_col=price_col)

    out_cols = [timestamp_col, price_col]
    if volume_col and volume_col in prepared.columns:
        out_cols.append(volume_col)

    prepared = prepared[out_cols].copy()

    rename_map: dict[str, str] = {}
    if timestamp_col != "timestamp":
        rename_map[timestamp_col] = "timestamp"
    if price_col != "price":
        rename_map[price_col] = "price"
    if volume_col and volume_col in prepared.columns and volume_col != "volume":
        rename_map[volume_col] = "volume"

    if rename_map:
        prepared = prepared.rename(columns=rename_map)

    if market_id is not None:
        prepared["market_id"] = market_id

    return prepared
