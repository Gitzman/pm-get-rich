"""DuckDB storage interface for Polymarket data."""

from pathlib import Path

import duckdb

from src.config import settings


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, creating the database directory if needed."""
    db_path = db_path or settings.duckdb_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def load_markets_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    sample: int | None = None,
) -> int:
    """Load markets.parquet into the markets table using DuckDB's native reader.

    Creates the table schema from the parquet file itself, preserving all columns.

    Args:
        con: DuckDB connection.
        parquet_path: Path to markets.parquet.
        sample: If set, load only the first N rows.

    Returns:
        Number of rows loaded.
    """
    limit = f"LIMIT {sample}" if sample else ""
    con.execute("DROP TABLE IF EXISTS markets")
    con.execute(f"""
        CREATE TABLE markets AS
        SELECT * FROM read_parquet('{parquet_path}')
        {limit}
    """)
    count = con.execute("SELECT count(*) FROM markets").fetchone()[0]
    return count


def load_trades_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    sample_market_ids: list[str] | None = None,
) -> int:
    """Load quant.parquet into the trades table using DuckDB's native reader.

    Creates the table schema from the parquet file itself, preserving all columns.

    Args:
        con: DuckDB connection.
        parquet_path: Path to quant.parquet.
        sample_market_ids: If set, only load trades for these market IDs.

    Returns:
        Number of rows loaded.
    """
    con.execute("DROP TABLE IF EXISTS trades")

    if sample_market_ids:
        ids_list = ", ".join(f"'{mid}'" for mid in sample_market_ids)
        con.execute(f"""
            CREATE TABLE trades AS
            SELECT * FROM read_parquet('{parquet_path}')
            WHERE market_id IN ({ids_list})
        """)
    else:
        con.execute(f"""
            CREATE TABLE trades AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

    count = con.execute("SELECT count(*) FROM trades").fetchone()[0]
    return count


def get_resolved_markets(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return all markets that have been resolved (closed with an outcome).

    The HF Polymarket dataset stores ``closed`` as UTINYINT (0/1) and has no
    ``outcome`` column.  Resolution is encoded in ``outcome_prices``, a JSON
    string like ``'["1","0"]'``.  The first element is the YES-token payout, so
    rounding it gives a binary 1 (YES) or 0 (NO) outcome.
    """
    rows = con.execute("""
        SELECT *,
               ROUND(CAST(
                   json_extract_string(outcome_prices, '$[0]') AS DOUBLE
               )) AS outcome
        FROM markets
        WHERE closed = 1
          AND outcome_prices IS NOT NULL
          AND outcome_prices != ''
        ORDER BY end_date DESC
    """).fetchall()
    columns = [desc[0] for desc in con.description]
    return [dict(zip(columns, row)) for row in rows]


def get_price_series(
    con: duckdb.DuckDBPyConnection, market_id: str
) -> list[dict]:
    """Return the price time series for a specific market."""
    rows = con.execute("""
        SELECT timestamp, price, usd_amount AS volume
        FROM trades
        WHERE market_id = ?
        ORDER BY timestamp ASC
    """, [market_id]).fetchall()
    columns = [desc[0] for desc in con.description]
    return [dict(zip(columns, row)) for row in rows]


def get_markets_with_min_datapoints(
    con: duckdb.DuckDBPyConnection, n: int
) -> list[dict]:
    """Return markets that have at least n price data points."""
    rows = con.execute("""
        SELECT m.*, t.datapoints
        FROM markets m
        JOIN (
            SELECT market_id, count(*) as datapoints
            FROM trades
            GROUP BY market_id
            HAVING count(*) >= ?
        ) t ON m.id = t.market_id
        ORDER BY t.datapoints DESC
    """, [n]).fetchall()
    columns = [desc[0] for desc in con.description]
    return [dict(zip(columns, row)) for row in rows]
