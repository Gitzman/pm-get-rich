"""Download Polymarket dataset from HuggingFace and load into DuckDB."""

import argparse
import sys
import time

from src.config import settings
from src.ingest.hf_loader import download_dataset
from src.store.db import (
    connect,
    load_markets_parquet,
    load_trades_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Polymarket data from HuggingFace and load into DuckDB."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Load only the first N markets (for dev/testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist locally.",
    )
    args = parser.parse_args()

    print(f"Dataset: {settings.hf_dataset}")
    print(f"DuckDB:  {settings.duckdb_path}")
    if args.sample:
        print(f"Sample:  first {args.sample} markets")
    print()

    # Step 1: Download parquet files
    print("=== Step 1: Downloading parquet files ===")
    t0 = time.monotonic()
    downloaded = download_dataset(force=args.force)
    elapsed = time.monotonic() - t0
    print(f"  Downloaded {len(downloaded)} files in {elapsed:.1f}s\n")

    # Resolve paths
    markets_path = None
    trades_path = None
    for name, path in downloaded.items():
        if "markets" in name.lower():
            markets_path = path
        elif "quant" in name.lower():
            trades_path = path

    if not markets_path:
        print("ERROR: markets.parquet not found in downloaded files", file=sys.stderr)
        sys.exit(1)
    if not trades_path:
        print("ERROR: quant.parquet not found in downloaded files", file=sys.stderr)
        sys.exit(1)

    # Step 2: Load into DuckDB
    print("=== Step 2: Loading into DuckDB ===")
    con = connect()

    print("  Loading markets...")
    t0 = time.monotonic()
    n_markets = load_markets_parquet(con, markets_path, sample=args.sample)
    elapsed = time.monotonic() - t0
    print(f"  ✓ {n_markets:,} markets loaded in {elapsed:.1f}s")

    # For --sample mode, only load trades for the sampled markets
    sample_ids = None
    if args.sample:
        rows = con.execute("SELECT id FROM markets").fetchall()
        sample_ids = [r[0] for r in rows]

    print("  Loading trades...")
    t0 = time.monotonic()
    n_trades = load_trades_parquet(con, trades_path, sample_market_ids=sample_ids)
    elapsed = time.monotonic() - t0
    print(f"  ✓ {n_trades:,} trades loaded in {elapsed:.1f}s")

    con.close()

    print(f"\nDone. Database at {settings.duckdb_path}")
    print(f"  Markets: {n_markets:,}")
    print(f"  Trades:  {n_trades:,}")


if __name__ == "__main__":
    main()
