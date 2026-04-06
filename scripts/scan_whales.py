"""Scan for whale activity on Polymarket contracts."""

from src.config import settings


def main() -> None:
    print(f"Scanning whales with min volume: {settings.min_volume}")
    print(f"Max markets: {settings.max_markets}")
    raise NotImplementedError("scan_whales not yet implemented")


if __name__ == "__main__":
    main()
