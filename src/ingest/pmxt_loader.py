"""PMXT L2 order book data loader.

Loads Polymarket order book snapshots from PMXT Parquet files hosted at
r2.pmxt.dev. Each file covers one hour of data across ALL markets; callers
filter by condition_id at scan time.

Schema per row:
  market_id   : str   — Polymarket condition_id (token ID)
  update_type : str   — "book_snapshot" or "price_change"
  data        : str   — JSON blob with book state

For book_snapshot, `data` contains:
  {
    "best_bid": float,
    "best_ask": float,
    "bids": [[price, size], ...],   # sorted desc by price
    "asks": [[price, size], ...],   # sorted asc by price
    "timestamp": int                # epoch millis
  }
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Where PMXT hourly Parquet files live.
PMXT_BASE_URL = "https://r2.pmxt.dev"

# Local cache directory for downloaded files.
PMXT_CACHE_DIR = Path(tempfile.gettempdir()) / "pmxt_cache"


@dataclass(frozen=True)
class BookLevel:
    """A single price level in the order book."""

    price: float
    size: float


@dataclass(frozen=True)
class BookSnapshot:
    """Full L2 order book state at a point in time."""

    timestamp: datetime
    market_id: str
    best_bid: float
    best_ask: float
    bids: list[BookLevel]  # sorted descending by price
    asks: list[BookLevel]  # sorted ascending by price

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def depth_at_price(self, price: float, side: str) -> float:
        """Total resting size at the given price level.

        Args:
            price: The price level to query.
            side: "bid" or "ask".

        Returns:
            Size at that price, or 0.0 if no level exists.
        """
        levels = self.bids if side == "bid" else self.asks
        for lvl in levels:
            if abs(lvl.price - price) < 1e-9:
                return lvl.size
        return 0.0

    def depth_at_or_better(self, price: float, side: str) -> float:
        """Total resting size at the given price or better (ahead in queue).

        For bids: levels at prices >= our price (higher bids fill first).
        For asks: levels at prices <= our price (lower asks fill first).
        """
        levels = self.bids if side == "bid" else self.asks
        total = 0.0
        for lvl in levels:
            if side == "bid" and lvl.price >= price:
                total += lvl.size
            elif side == "ask" and lvl.price <= price:
                total += lvl.size
        return total


def _parquet_filename(dt: datetime) -> str:
    """Generate the PMXT parquet filename for a given hour.

    Format: polymarket_orderbook_YYYY-MM-DDTHH.parquet
    """
    return f"polymarket_orderbook_{dt.strftime('%Y-%m-%dT%H')}.parquet"


def _download_parquet(dt: datetime, cache_dir: Path | None = None) -> Path:
    """Download a PMXT hourly parquet file, caching locally.

    Args:
        dt: Datetime identifying the hour to download.
        cache_dir: Directory for local cache. Defaults to PMXT_CACHE_DIR.

    Returns:
        Path to the local parquet file.
    """
    import httpx

    cache = cache_dir or PMXT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    filename = _parquet_filename(dt)
    local_path = cache / filename

    if local_path.exists():
        return local_path

    url = f"{PMXT_BASE_URL}/{filename}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)

    return local_path


def parse_book_snapshot(row: dict) -> BookSnapshot | None:
    """Parse a single PMXT row into a BookSnapshot.

    Args:
        row: Dict with market_id, update_type, data fields.

    Returns:
        BookSnapshot if update_type is book_snapshot and data parses, else None.
    """
    if row.get("update_type") != "book_snapshot":
        return None

    data = row.get("data", "{}")
    if isinstance(data, str):
        data = json.loads(data)

    ts_ms = data.get("timestamp", 0)
    ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc) if ts_ms else None
    if ts is None:
        return None

    bids = [BookLevel(price=float(b[0]), size=float(b[1])) for b in data.get("bids", [])]
    asks = [BookLevel(price=float(a[0]), size=float(a[1])) for a in data.get("asks", [])]

    # Ensure sorting: bids desc by price, asks asc by price
    bids.sort(key=lambda l: l.price, reverse=True)
    asks.sort(key=lambda l: l.price)

    raw_bid = data.get("best_bid")
    raw_ask = data.get("best_ask")
    best_bid = float(raw_bid) if raw_bid is not None else (bids[0].price if bids else 0.0)
    best_ask = float(raw_ask) if raw_ask is not None else (asks[0].price if asks else 1.0)

    return BookSnapshot(
        timestamp=ts,
        market_id=row["market_id"],
        best_bid=best_bid,
        best_ask=best_ask,
        bids=bids,
        asks=asks,
    )


def load_book_snapshots(
    start: datetime,
    end: datetime,
    condition_ids: list[str],
    cache_dir: Path | None = None,
) -> list[BookSnapshot]:
    """Load order book snapshots for specific markets over a time range.

    Downloads hourly PMXT parquet files covering [start, end], filters to
    the given condition_ids, and returns parsed BookSnapshots sorted by time.

    Args:
        start: Start of the time range (UTC).
        end: End of the time range (UTC).
        condition_ids: List of Polymarket condition_ids to filter for.
        cache_dir: Optional cache directory override.

    Returns:
        List of BookSnapshot sorted by timestamp.
    """
    # Generate list of hours to cover
    from datetime import timedelta

    current = start.replace(minute=0, second=0, microsecond=0)
    hours: list[datetime] = []
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)

    condition_set = set(condition_ids)
    snapshots: list[BookSnapshot] = []

    for hour_dt in hours:
        try:
            parquet_path = _download_parquet(hour_dt, cache_dir)
        except Exception:
            # File may not exist for this hour; skip.
            continue

        df = pl.read_parquet(parquet_path)

        # Filter to our condition_ids and book_snapshot type
        filtered = df.filter(
            (pl.col("market_id").is_in(list(condition_set)))
            & (pl.col("update_type") == "book_snapshot")
        )

        for row in filtered.iter_rows(named=True):
            snap = parse_book_snapshot(row)
            if snap is not None:
                snapshots.append(snap)

    snapshots.sort(key=lambda s: s.timestamp)
    return snapshots


def load_book_snapshots_from_parquet(
    parquet_path: Path | str,
    condition_ids: list[str] | None = None,
) -> list[BookSnapshot]:
    """Load book snapshots from a local PMXT parquet file.

    Useful for testing or when files are already downloaded.

    Args:
        parquet_path: Path to a local PMXT parquet file.
        condition_ids: Optional list of condition_ids to filter.
            If None, loads all markets in the file.

    Returns:
        List of BookSnapshot sorted by timestamp.
    """
    df = pl.read_parquet(str(parquet_path))

    if condition_ids:
        df = df.filter(pl.col("market_id").is_in(condition_ids))

    df = df.filter(pl.col("update_type") == "book_snapshot")

    snapshots: list[BookSnapshot] = []
    for row in df.iter_rows(named=True):
        snap = parse_book_snapshot(row)
        if snap is not None:
            snapshots.append(snap)

    snapshots.sort(key=lambda s: s.timestamp)
    return snapshots
