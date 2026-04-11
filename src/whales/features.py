"""Whale-based feature extraction for time series forecasting.

Provides functions to identify top/bottom profitable wallets and
compute whale-derived trade features per market.
"""

from __future__ import annotations

import duckdb


def get_whale_addresses(
    con: duckdb.DuckDBPyConnection,
    top_n: int = 20,
    bottom_n: int = 20,
    min_markets: int = 5,
) -> tuple[list[str], list[str]]:
    """Rank wallets by PnL across all resolved markets and return top/bottom addresses.

    Uses both taker and maker PnL from the trades table, matching the
    whale leaderboard query used elsewhere in the codebase.

    Args:
        con: DuckDB connection with trades and markets tables loaded.
        top_n: Number of top-performing wallet addresses to return.
        bottom_n: Number of bottom-performing wallet addresses to return.
        min_markets: Minimum distinct markets traded to qualify.

    Returns:
        Tuple of (top_addresses, bottom_addresses) sorted by PnL descending
        and ascending respectively.
    """
    rows = con.execute(
        """
        WITH resolved AS (
            SELECT id,
                   ROUND(CAST(
                       json_extract_string(outcome_prices, '$[0]') AS DOUBLE
                   )) AS outcome
            FROM markets
            WHERE closed = 1
              AND outcome_prices IS NOT NULL
              AND outcome_prices != ''
        ),
        taker_pnl AS (
            SELECT
                t.taker AS addr, t.market_id, t.usd_amount,
                CASE
                    WHEN t.side='BUY'  AND r.outcome=1
                        THEN t.usd_amount * (1.0 / t.price - 1.0)
                    WHEN t.side='BUY'  AND r.outcome=0
                        THEN -t.usd_amount
                    WHEN t.side='SELL' AND r.outcome=0
                        THEN t.usd_amount
                    WHEN t.side='SELL' AND r.outcome=1
                        THEN -t.usd_amount * (1.0 / t.price - 1.0)
                END AS pnl
            FROM trades t
            JOIN resolved r ON t.market_id = r.id
            WHERE t.price > 0 AND t.price < 1
        ),
        maker_pnl AS (
            SELECT
                t.maker AS addr, t.market_id, t.usd_amount,
                CASE
                    WHEN t.side='BUY'  AND r.outcome=0
                        THEN t.usd_amount
                    WHEN t.side='BUY'  AND r.outcome=1
                        THEN -t.usd_amount * (1.0 / t.price - 1.0)
                    WHEN t.side='SELL' AND r.outcome=1
                        THEN t.usd_amount * (1.0 / t.price - 1.0)
                    WHEN t.side='SELL' AND r.outcome=0
                        THEN -t.usd_amount
                END AS pnl
            FROM trades t
            JOIN resolved r ON t.market_id = r.id
            WHERE t.price > 0 AND t.price < 1
        ),
        all_pnl AS (
            SELECT * FROM taker_pnl
            UNION ALL
            SELECT * FROM maker_pnl
        ),
        ranked AS (
            SELECT
                addr,
                count(DISTINCT market_id) AS n_markets,
                sum(pnl) AS total_pnl
            FROM all_pnl
            GROUP BY addr
            HAVING count(DISTINCT market_id) >= ?
        )
        SELECT addr, total_pnl
        FROM ranked
        ORDER BY total_pnl DESC
        """,
        [min_markets],
    ).fetchall()

    top_addrs = [r[0] for r in rows[:top_n]]
    bottom_addrs = [r[0] for r in rows[-bottom_n:]] if len(rows) >= bottom_n else [r[0] for r in rows]

    return top_addrs, bottom_addrs
