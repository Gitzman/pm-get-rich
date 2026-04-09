"""Normalize weather temperature events into the shared event schema.

Reads from the shared DuckDB, produces:
  - data/events/<event_id>/events.parquet  (partitioned Parquet)
  - data/events/<event_id>/events.jsonl    (JSONL for inspection)
  - data/events/<event_id>/_meta.json      (event metadata)

Usage:
    uv run python scripts/normalize_events.py [--db PATH] [--top N] [--event EVENT_ID]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import polars as pl


# ---------------------------------------------------------------------------
# Suit extraction
# ---------------------------------------------------------------------------

# Celsius single-degree: "be 9°C on"
_CITY_TEMP_EXACT_C = re.compile(r"be (\d+)°C on")
_CITY_TEMP_LOW_C = re.compile(r"be (\d+)°C or below")
_CITY_TEMP_HIGH_C = re.compile(r"be (\d+)°C or higher")

# Fahrenheit range: "between 56-57°F on"
_CITY_TEMP_RANGE_F = re.compile(r"between (\d+)-(\d+)°F on")
_CITY_TEMP_LOW_F = re.compile(r"be (\d+)°F or below")
_CITY_TEMP_HIGH_F = re.compile(r"be (\d+)°F or higher")

# Temperature increase (supports both °C and ºC)
_INCREASE_RANGE_AND = re.compile(r"between ([\d.]+)[°º]C and ([\d.]+)[°º]C")
_INCREASE_RANGE_DASH = re.compile(r"between ([\d.]+)-([\d.]+)[°º]C")
_INCREASE_LESS = re.compile(r"less than ([\d.]+)[°º]C")
_INCREASE_MORE = re.compile(r"(?:more|greater) than ([\d.]+)[°º]C")


def extract_suit(question: str) -> str:
    """Extract bucket label (suit) from a market question."""
    # Celsius single-degree
    m = _CITY_TEMP_LOW_C.search(question)
    if m:
        return f"<={m.group(1)}C"
    m = _CITY_TEMP_HIGH_C.search(question)
    if m:
        return f">={m.group(1)}C"
    m = _CITY_TEMP_EXACT_C.search(question)
    if m:
        return f"{m.group(1)}C"
    # Fahrenheit range
    m = _CITY_TEMP_LOW_F.search(question)
    if m:
        return f"<={m.group(1)}F"
    m = _CITY_TEMP_HIGH_F.search(question)
    if m:
        return f">={m.group(1)}F"
    m = _CITY_TEMP_RANGE_F.search(question)
    if m:
        return f"{m.group(1)}-{m.group(2)}F"
    # Temperature increase ranges
    m = _INCREASE_RANGE_AND.search(question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _INCREASE_RANGE_DASH.search(question)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = _INCREASE_LESS.search(question)
    if m:
        return f"<{m.group(1)}"
    m = _INCREASE_MORE.search(question)
    if m:
        return f">{m.group(1)}"
    return question  # fallback: keep raw question


# ---------------------------------------------------------------------------
# Meta extraction
# ---------------------------------------------------------------------------

_CITY_DATE_RE = re.compile(
    r"Highest temperature in (.+?) on (.+)\?"
)
_TEMP_INCREASE_RE = re.compile(
    r"(\w+ \d{4}) Temperature Increase"
)


def extract_meta(event_title: str) -> dict:
    """Extract city and date from event title."""
    m = _CITY_DATE_RE.search(event_title)
    if m:
        return {"city": m.group(1), "date": m.group(2)}
    m = _TEMP_INCREASE_RE.search(event_title)
    if m:
        return {"city": "global", "date": m.group(1)}
    return {"city": "unknown", "date": "unknown"}


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def get_top_temperature_events(
    con: duckdb.DuckDBPyConnection, top_n: int
) -> list[dict]:
    """Return top N temperature events by trade count."""
    rows = con.execute("""
        SELECT
            wm.event_id,
            wm.event_title,
            COUNT(*) as n_trades
        FROM weather_markets wm
        JOIN trades t ON t.market_id = wm.id
        WHERE LOWER(wm.event_title) LIKE '%temperature%'
        GROUP BY wm.event_id, wm.event_title
        ORDER BY n_trades DESC
        LIMIT ?
    """, [top_n]).fetchall()
    return [{"event_id": r[0], "event_title": r[1], "n_trades": r[2]} for r in rows]


def normalize_event(
    con: duckdb.DuckDBPyConnection,
    event_id: str,
    output_dir: Path,
) -> dict:
    """Normalize a single event into the shared schema.

    Returns metadata dict for the event.
    """
    # Get markets for this event
    markets = con.execute("""
        SELECT id, question, outcome_prices
        FROM weather_markets
        WHERE event_id = ?
    """, [event_id]).fetchall()

    if not markets:
        raise ValueError(f"No markets found for event_id={event_id}")

    # Build market_id -> suit mapping and find outcome
    suit_map: dict[str, str] = {}
    outcome_bucket = None
    for mid, question, outcome_prices in markets:
        suit = extract_suit(question)
        suit_map[mid] = suit
        # Determine outcome: parse python-style list string
        if outcome_prices:
            prices = outcome_prices.strip("[]").replace("'", "").split(",")
            if len(prices) >= 1:
                try:
                    yes_price = float(prices[0].strip())
                    if yes_price > 0.5:
                        outcome_bucket = suit
                except ValueError:
                    pass

    # Get resolved outcome from weather_resolved if available
    resolved = con.execute("""
        SELECT question FROM weather_resolved
        WHERE event_id = ? AND outcome = 1.0
        LIMIT 1
    """, [event_id]).fetchone()
    if resolved:
        outcome_bucket = extract_suit(resolved[0])

    # Get event title
    event_meta = con.execute("""
        SELECT DISTINCT event_title FROM weather_markets WHERE event_id = ?
    """, [event_id]).fetchone()
    event_title = event_meta[0] if event_meta else "unknown"

    # Get all trades for this event, deduplicated, ordered for seq assignment
    trades_df = con.execute("""
        SELECT DISTINCT ON (t.block_number, t.log_index, t.market_id)
            t.event_id,
            t.block_number,
            t.log_index,
            t.timestamp,
            t.market_id,
            t.price,
            t.usd_amount,
            t.token_amount,
            t.side,
            t.taker,
            t.maker,
            t.transaction_hash
        FROM trades t
        WHERE t.event_id = ?
        ORDER BY t.block_number, t.log_index, t.market_id
    """, [event_id]).pl()

    if trades_df.is_empty():
        raise ValueError(f"No trades found for event_id={event_id}")

    n_rows = trades_df.height

    # Build the normalized dataframe using select (avoids Expr-in-constructor issues)
    _ob = outcome_bucket or "unknown"
    normalized = trades_df.select(
        pl.col("event_id").cast(pl.Utf8),
        pl.lit(event_title).alias("event_title"),
        pl.arange(0, pl.len()).cast(pl.Int64).alias("seq"),
        (pl.col("timestamp").cast(pl.Int64) * 1000).alias("timestamp_ms"),
        pl.col("block_number").cast(pl.Int64),
        pl.col("log_index").cast(pl.Int64),
        pl.lit("trade").alias("event_type"),
        pl.col("taker").alias("actor"),
        pl.lit("taker").alias("actor_role"),
        pl.col("side"),
        pl.col("market_id"),
        pl.col("market_id")
        .map_elements(lambda mid: suit_map.get(mid, mid), return_dtype=pl.Utf8)
        .alias("suit"),
        pl.col("price"),
        pl.col("token_amount").alias("size"),
        pl.col("usd_amount"),
        (pl.col("transaction_hash") + pl.lit(":") + pl.col("log_index").cast(pl.Utf8))
        .alias("fill_id"),
        pl.col("maker").alias("counterparty"),
        pl.lit("maker").alias("counterparty_role"),
        pl.col("market_id")
        .map_elements(
            lambda mid: 1.0 if suit_map.get(mid, mid) == _ob else 0.0,
            return_dtype=pl.Float64,
        )
        .alias("outcome"),
        pl.lit(_ob).alias("outcome_bucket"),
    )

    # Extract meta info
    meta_info = extract_meta(event_title)
    bucket_labels = sorted(set(suit_map.values()))
    ts_min = int(trades_df["timestamp"].min())
    ts_max = int(trades_df["timestamp"].max())

    meta = {
        "event_id": event_id,
        "event_title": event_title,
        "city": meta_info["city"],
        "date": meta_info["date"],
        "n_buckets": len(bucket_labels),
        "bucket_labels": bucket_labels,
        "outcome_bucket": outcome_bucket or "unknown",
        "time_range": {
            "start_epoch_s": ts_min,
            "end_epoch_s": ts_max,
        },
        "n_trades": n_rows,
        "n_markets": len(markets),
    }

    # Write outputs
    event_dir = output_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    # Parquet
    normalized.write_parquet(event_dir / "events.parquet")

    # JSONL
    with open(event_dir / "events.jsonl", "w") as f:
        for row in normalized.iter_rows(named=True):
            f.write(json.dumps(row, default=str) + "\n")

    # Meta
    with open(event_dir / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_db() -> Path:
    """Locate the shared DuckDB. Check local data/ first, then mayor/rig/data/."""
    local = Path("data/pmgetrich.duckdb")
    if local.exists():
        return local
    # Walk up to find the rig root
    for candidate in [
        Path("/home/gitzman/gt/pmgetrich/mayor/rig/data/pmgetrich.duckdb"),
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find pmgetrich.duckdb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize weather events")
    parser.add_argument("--db", type=Path, help="Path to DuckDB file")
    parser.add_argument("--top", type=int, default=20, help="Top N events")
    parser.add_argument("--event", type=str, help="Single event ID to process")
    parser.add_argument(
        "--out", type=Path, default=Path("data/events"), help="Output directory"
    )
    args = parser.parse_args()

    db_path = args.db or find_db()
    print(f"Using database: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)

    if args.event:
        events = [{"event_id": args.event}]
    else:
        events = get_top_temperature_events(con, args.top)
        print(f"Found {len(events)} temperature events to normalize")

    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also write a combined parquet partitioned by event_id
    all_frames: list[pl.DataFrame] = []

    for i, evt in enumerate(events):
        eid = evt["event_id"]
        title = evt.get("event_title", "")
        print(f"  [{i+1}/{len(events)}] event={eid} {title[:60]}")
        try:
            meta = normalize_event(con, eid, output_dir)
            print(
                f"    -> {meta['n_trades']} trades, "
                f"{meta['n_buckets']} buckets, "
                f"outcome={meta['outcome_bucket']}"
            )
            # Read back for combined
            df = pl.read_parquet(output_dir / eid / "events.parquet")
            all_frames.append(df)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)

    # Write combined partitioned parquet
    if all_frames:
        combined = pl.concat(all_frames)
        combined.write_parquet(
            output_dir / "all_events.parquet",
            use_pyarrow=True,
        )
        print(f"\nCombined: {combined.height} total trades across {len(all_frames)} events")
        print(f"Output: {output_dir}/")

    con.close()


if __name__ == "__main__":
    main()
