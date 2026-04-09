"""Export DuckDB tables to Parquet for the viz static site."""

from pathlib import Path

import duckdb

from src.store.db import connect


def main() -> None:
    out_dir = Path("viz/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect()

    # 1. Weather resolved markets
    con.execute(f"""
        COPY weather_resolved
        TO '{out_dir}/weather_resolved.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute("SELECT count(*) FROM weather_resolved").fetchone()[0]
    print(f"Exported weather_resolved: {n} rows")

    # 2. All weather markets (including unresolved)
    con.execute(f"""
        COPY weather_markets
        TO '{out_dir}/weather_markets.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute("SELECT count(*) FROM weather_markets").fetchone()[0]
    print(f"Exported weather_markets: {n} rows")

    # 3. Whale leaderboard (aggregated PnL by trader address)
    con.execute(f"""
        COPY (
            WITH taker_pnl AS (
                SELECT
                    t.taker AS address, t.market_id, t.usd_amount,
                    CASE
                        WHEN t.side='BUY'  AND w.outcome=1 THEN t.usd_amount * (1.0/t.price - 1.0)
                        WHEN t.side='BUY'  AND w.outcome=0 THEN -t.usd_amount
                        WHEN t.side='SELL' AND w.outcome=0 THEN t.usd_amount
                        WHEN t.side='SELL' AND w.outcome=1 THEN -t.usd_amount * (1.0/t.price - 1.0)
                    END AS pnl
                FROM trades t
                JOIN weather_resolved w ON t.market_id = w.id
                WHERE t.price > 0 AND t.price < 1
            ),
            maker_pnl AS (
                SELECT
                    t.maker AS address, t.market_id, t.usd_amount,
                    CASE
                        WHEN t.side='BUY'  AND w.outcome=0 THEN t.usd_amount
                        WHEN t.side='BUY'  AND w.outcome=1 THEN -t.usd_amount * (1.0/t.price - 1.0)
                        WHEN t.side='SELL' AND w.outcome=1 THEN t.usd_amount * (1.0/t.price - 1.0)
                        WHEN t.side='SELL' AND w.outcome=0 THEN -t.usd_amount
                    END AS pnl
                FROM trades t
                JOIN weather_resolved w ON t.market_id = w.id
                WHERE t.price > 0 AND t.price < 1
            ),
            all_pnl AS (
                SELECT * FROM taker_pnl
                UNION ALL
                SELECT * FROM maker_pnl
            )
            SELECT
                address,
                count(DISTINCT market_id) AS n_markets,
                sum(pnl) AS total_pnl,
                sum(usd_amount) AS total_wagered,
                sum(pnl) / NULLIF(sum(usd_amount), 0) AS roi
            FROM all_pnl
            GROUP BY address
            HAVING count(DISTINCT market_id) >= 5
            ORDER BY total_pnl DESC
        ) TO '{out_dir}/whale_leaderboard.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_dir}/whale_leaderboard.parquet')"
    ).fetchone()[0]
    print(f"Exported whale_leaderboard: {n} rows")

    con.close()

    total = sum(f.stat().st_size for f in out_dir.glob("*.parquet"))
    print(f"\nTotal: {total / 1024:.0f} KB in {out_dir}/")


if __name__ == "__main__":
    main()
