"""Scan for divergence signals between forecast and market prices."""

from src.config import settings


def main() -> None:
    print(f"Scanning signals with threshold: {settings.divergence_threshold}")
    print(f"Lookback: {settings.divergence_lookback_days} days")
    raise NotImplementedError("scan_signals not yet implemented")


if __name__ == "__main__":
    main()
