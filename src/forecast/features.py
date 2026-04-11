from __future__ import annotations

import duckdb
import numpy as np
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


# ---------------------------------------------------------------------------
# Trade-activity feature computation
# ---------------------------------------------------------------------------

_FREQ_TO_DUCKDB_BUCKET = {
    "1h": "hour",
    "1D": "day",
    "1d": "day",
}


def _bucket_sql(freq: str) -> str:
    """Convert pandas freq string to DuckDB date_trunc unit."""
    unit = _FREQ_TO_DUCKDB_BUCKET.get(freq)
    if unit is None:
        raise ValueError(
            f"Unsupported freq '{freq}'. Use one of: {list(_FREQ_TO_DUCKDB_BUCKET)}"
        )
    return unit


def compute_trade_features(
    con: duckdb.DuckDBPyConnection,
    market_id: str,
    freq: str = "1h",
    top_whales: list[str] | None = None,
    bottom_whales: list[str] | None = None,
    zscore_window: int = 24,
) -> pd.DataFrame:
    """Compute per-bucket trade-activity features for a single market.

    Returns a DataFrame indexed by bucket timestamp with columns:
        - volume: total USD volume per bucket
        - volume_zscore: volume z-score relative to rolling mean/std
        - maker_taker_imbalance: (maker_volume - taker_volume) / total_volume
        - whale_net_flow: net USD from top whale wallets (positive=buying)
        - smart_dumb_ratio: top-whale volume / bottom-whale volume

    Args:
        con: DuckDB connection with trades table loaded.
        market_id: Market to compute features for.
        freq: Time bucket frequency ('1h' or '1D').
        top_whales: Addresses of top-performing wallets. If None, whale
            features are filled with 0.
        bottom_whales: Addresses of bottom-performing wallets. If None,
            smart_dumb_ratio is filled with 0.
        zscore_window: Rolling window size (in buckets) for volume z-score.
    """
    bucket_unit = _bucket_sql(freq)

    # Base volume and maker/taker aggregation per bucket
    base_df = con.execute(
        """
        SELECT
            date_trunc(?, epoch_ms(timestamp * 1000)) AS bucket,
            sum(usd_amount) AS volume,
            sum(CASE WHEN side = 'BUY' THEN usd_amount ELSE -usd_amount END)
                AS net_buy_volume,
            -- Maker flow: maker provides liquidity. For BUY trades maker is
            -- on the sell side; for SELL trades maker is on the buy side.
            sum(CASE WHEN side = 'BUY' THEN -usd_amount ELSE usd_amount END)
                AS maker_flow,
            sum(CASE WHEN side = 'BUY' THEN usd_amount ELSE -usd_amount END)
                AS taker_flow
        FROM trades
        WHERE market_id = ?
          AND price > 0 AND price < 1
        GROUP BY bucket
        ORDER BY bucket
        """,
        [bucket_unit, market_id],
    ).fetchdf()

    if base_df.empty:
        return pd.DataFrame()

    base_df = base_df.rename(columns={"bucket": "timestamp"})

    # Volume z-score: rolling mean/std
    rolling_mean = base_df["volume"].rolling(window=zscore_window, min_periods=1).mean()
    rolling_std = base_df["volume"].rolling(window=zscore_window, min_periods=1).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    base_df["volume_zscore"] = ((base_df["volume"] - rolling_mean) / rolling_std).fillna(0.0)

    # Maker-taker imbalance: normalized to [-1, 1]
    total_vol = base_df["volume"].replace(0, np.nan)
    base_df["maker_taker_imbalance"] = (
        (base_df["maker_flow"] - base_df["taker_flow"]) / total_vol
    ).fillna(0.0)

    # Whale features (require address lists)
    if top_whales:
        whale_df = _compute_whale_flow(con, market_id, bucket_unit, top_whales)
        base_df = base_df.merge(whale_df, on="timestamp", how="left")
        base_df["whale_net_flow"] = base_df["whale_net_flow"].fillna(0.0)
    else:
        base_df["whale_net_flow"] = 0.0

    if top_whales and bottom_whales:
        smart_vol = _compute_address_volume(con, market_id, bucket_unit, top_whales)
        dumb_vol = _compute_address_volume(con, market_id, bucket_unit, bottom_whales)
        merged = smart_vol.merge(
            dumb_vol, on="timestamp", how="outer", suffixes=("_smart", "_dumb")
        ).fillna(0.0)
        denom = merged["addr_volume_dumb"].replace(0, np.nan)
        merged["smart_dumb_ratio"] = (merged["addr_volume_smart"] / denom).fillna(0.0)
        base_df = base_df.merge(
            merged[["timestamp", "smart_dumb_ratio"]], on="timestamp", how="left"
        )
        base_df["smart_dumb_ratio"] = base_df["smart_dumb_ratio"].fillna(0.0)
    else:
        base_df["smart_dumb_ratio"] = 0.0

    # Drop intermediate columns
    keep_cols = [
        "timestamp", "volume", "volume_zscore",
        "maker_taker_imbalance", "whale_net_flow", "smart_dumb_ratio",
    ]
    return base_df[[c for c in keep_cols if c in base_df.columns]]


def _compute_whale_flow(
    con: duckdb.DuckDBPyConnection,
    market_id: str,
    bucket_unit: str,
    whale_addrs: list[str],
) -> pd.DataFrame:
    """Net USD flow from whale addresses per bucket.

    Positive = net buying by whales, negative = net selling.
    """
    placeholders = ",".join(["?"] * len(whale_addrs))
    params = [bucket_unit, market_id] + whale_addrs + whale_addrs

    df = con.execute(
        f"""
        WITH whale_trades AS (
            SELECT
                date_trunc(?, epoch_ms(timestamp * 1000)) AS bucket,
                CASE WHEN side = 'BUY' THEN usd_amount ELSE -usd_amount END
                    AS signed_amount
            FROM trades
            WHERE market_id = ?
              AND price > 0 AND price < 1
              AND (taker IN ({placeholders}) OR maker IN ({placeholders}))
        )
        SELECT bucket AS timestamp, sum(signed_amount) AS whale_net_flow
        FROM whale_trades
        GROUP BY bucket
        ORDER BY bucket
        """,
        params,
    ).fetchdf()

    return df


def _compute_address_volume(
    con: duckdb.DuckDBPyConnection,
    market_id: str,
    bucket_unit: str,
    addresses: list[str],
) -> pd.DataFrame:
    """Total USD volume from a set of addresses per bucket."""
    placeholders = ",".join(["?"] * len(addresses))
    params = [bucket_unit, market_id] + addresses + addresses

    df = con.execute(
        f"""
        SELECT
            date_trunc(?, epoch_ms(timestamp * 1000)) AS timestamp,
            sum(usd_amount) AS addr_volume
        FROM trades
        WHERE market_id = ?
          AND price > 0 AND price < 1
          AND (taker IN ({placeholders}) OR maker IN ({placeholders}))
        GROUP BY 1
        ORDER BY 1
        """,
        params,
    ).fetchdf()

    return df


def prepare_multivariate_series(
    con: duckdb.DuckDBPyConnection,
    market_id: str,
    freq: str = "1h",
    top_whales: list[str] | None = None,
    bottom_whales: list[str] | None = None,
    zscore_window: int = 24,
) -> pd.DataFrame:
    """Full preparation pipeline for multivariate Chronos input.

    Combines price series with trade-activity features into a single
    DataFrame suitable for Chronos 2 multivariate forecasting.

    Returns DataFrame with columns:
        - timestamp: uniform time buckets
        - price: YES-token price [0, 1]
        - market_id: market identifier
        - volume_zscore: trade volume z-score
        - maker_taker_imbalance: normalized maker vs taker flow
        - whale_net_flow: net USD from top whale wallets
        - smart_dumb_ratio: top-whale / bottom-whale volume ratio

    Args:
        con: DuckDB connection with trades and markets tables.
        market_id: Market identifier.
        freq: Time bucket frequency.
        top_whales: Top whale wallet addresses.
        bottom_whales: Bottom whale wallet addresses.
        zscore_window: Rolling window for volume z-score.
    """
    from src.store.db import get_price_series

    # Get price series
    raw_prices = get_price_series(con, market_id)
    if not raw_prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(raw_prices)
    price_prepared = prepare_series(
        price_df,
        freq=freq,
        timestamp_col="timestamp",
        price_col="price",
        market_id=market_id,
        volume_col="volume",
    )

    # Compute trade-activity features
    trade_features = compute_trade_features(
        con,
        market_id,
        freq=freq,
        top_whales=top_whales,
        bottom_whales=bottom_whales,
        zscore_window=zscore_window,
    )

    if trade_features.empty:
        # No trade features available, return price-only
        for col in ["volume_zscore", "maker_taker_imbalance",
                     "whale_net_flow", "smart_dumb_ratio"]:
            price_prepared[col] = 0.0
        return price_prepared

    # Merge on timestamp
    merged = price_prepared.merge(trade_features, on="timestamp", how="left")

    # Fill NaN features with 0 (buckets with price data but no trades)
    feature_cols = [
        "volume_zscore", "maker_taker_imbalance",
        "whale_net_flow", "smart_dumb_ratio",
    ]
    for col in feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0

    return merged
