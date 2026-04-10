"""Polymarket fee model.

Fee formula (taker): fee = θ × C × p × (1 - p)
  - θ (theta) varies by market category
  - C = number of contracts (shares)
  - p = limit price in [0.01, 0.99]
  - Fee is symmetric around p=0.50, lowest at extremes

Makers pay zero fees. Makers receive a daily USDC rebate equal to their
pro-rata share of 25% of taker fees collected in that market (20% for crypto).

Sources:
  - docs.polymarket.com/trading/fees (verified 2026-04-10)
  - polymarketexchange.com/fees-hours.html (verified 2026-04-10)
  - API: GET https://clob.polymarket.com/fee-rate?token_id={id}
"""

from __future__ import annotations

from dataclasses import dataclass

# Theta by market category (taker fee coefficient).
# Source: docs.polymarket.com/trading/fees, verified 2026-04-10.
CATEGORY_THETA: dict[str, float] = {
    "crypto": 0.072,
    "sports": 0.030,
    "finance": 0.040,
    "politics": 0.040,
    "economics": 0.050,
    "culture": 0.050,
    "weather": 0.050,
    "other": 0.050,
    "mentions": 0.040,
    "tech": 0.040,
    "geopolitics": 0.000,  # fee-free
}


@dataclass(frozen=True)
class FeeSchedule:
    """Fee parameters for a market category."""

    theta: float  # taker fee coefficient
    maker_rebate_pct: float  # fraction of taker fees rebated to makers (0.25 = 25%)
    tick: float = 0.01  # minimum price increment
    price_min: float = 0.01
    price_max: float = 0.99
    gas_cost_usd: float = 0.0  # Polygon meta-tx: effectively zero


# Default schedule for weather markets (our target category).
WEATHER_FEES = FeeSchedule(theta=0.050, maker_rebate_pct=0.25)


def taker_fee(price: float, contracts: float, theta: float = 0.050) -> float:
    """Compute the taker fee for a trade.

    Args:
        price: Limit price in [0.01, 0.99].
        contracts: Number of contracts (shares) traded.
        theta: Fee coefficient for the market category.

    Returns:
        Fee in USDC. Rounded to 5 decimal places per Polymarket spec.
    """
    fee = theta * contracts * price * (1.0 - price)
    return round(fee, 5)


def maker_rebate_share(
    price: float,
    contracts: float,
    theta: float = 0.050,
    rebate_pct: float = 0.25,
) -> float:
    """Upper-bound estimate of maker rebate for a single fill.

    The actual rebate is pooled daily across all makers in a market and
    distributed pro-rata by fee-equivalent volume. This function returns
    the theoretical maximum if you were the only maker — useful as an
    optimistic bound in cost modeling.

    Args:
        price: Fill price in [0.01, 0.99].
        contracts: Number of contracts filled.
        theta: Taker fee coefficient.
        rebate_pct: Fraction of taker fees in the rebate pool (0.25 for weather).

    Returns:
        Optimistic rebate estimate in USDC.
    """
    equivalent_taker_fee = theta * contracts * price * (1.0 - price)
    return round(equivalent_taker_fee * rebate_pct, 5)


def round_trip_cost(
    entry_price: float,
    exit_price: float,
    contracts: float,
    theta: float = 0.050,
    entry_is_maker: bool = True,
    exit_is_maker: bool = False,
    include_rebate: bool = False,
    rebate_pct: float = 0.25,
) -> float:
    """Total fee cost for entering and exiting a position.

    In our maker strategy, we enter via limit order (maker, zero fee) and
    exit when the market resolves or we cross the spread (taker fee).

    Args:
        entry_price: Price at which we enter the position.
        exit_price: Price at which we exit (or 1.0/0.0 at resolution).
        contracts: Number of contracts.
        theta: Fee coefficient.
        entry_is_maker: Whether entry is a maker order (no fee).
        exit_is_maker: Whether exit is a maker order (no fee).
        include_rebate: Whether to subtract optimistic rebate estimate.
        rebate_pct: Maker rebate fraction.

    Returns:
        Total round-trip fee cost in USDC (positive = cost).
    """
    cost = 0.0

    if not entry_is_maker:
        cost += taker_fee(entry_price, contracts, theta)
    if not exit_is_maker:
        cost += taker_fee(exit_price, contracts, theta)

    if include_rebate and entry_is_maker:
        cost -= maker_rebate_share(entry_price, contracts, theta, rebate_pct)

    return round(cost, 5)
