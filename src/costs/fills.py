"""Fill model for limit orders on Polymarket.

Assumptions:
  - We place maker (limit) orders and wait for taker flow to fill us.
  - A limit order at price P fills only if a taker trades at P or better
    within a time window M.
  - Queue position: conservative — we assume we are LAST at our price level.
    If N contracts are already resting at price P, all N must fill before ours.
  - Adverse selection: if we get filled, the price may have moved against us.
    We measure this as post-fill price drift over 30s and 60s windows.

The fill model is intentionally conservative. Overestimating fills leads to
phantom P&L; underestimating just understates returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FillAssumptions:
    """Parameters governing fill probability and adverse selection."""

    # Time window (seconds) within which a taker must arrive for a fill.
    fill_window_s: float = 300.0  # 5 minutes

    # Queue position assumption: fraction of resting depth we must wait behind.
    # 1.0 = pessimistic (last in queue), 0.5 = mid-queue, 0.0 = front of queue.
    queue_position_frac: float = 1.0

    # Minimum historical taker volume at our price level for a fill to be
    # considered plausible. Below this, fill_probability returns 0.
    min_taker_volume: float = 10.0  # contracts

    # Post-fill drift windows for adverse selection measurement (seconds).
    adverse_selection_windows: tuple[float, ...] = (30.0, 60.0)

    # Adverse selection penalty: expected price drift against us per fill,
    # expressed as a fraction of the tick size. Conservative default based on
    # market microstructure literature for thin order books.
    # 0.5 ticks at $0.01/tick = $0.005 per contract per fill.
    adverse_selection_ticks: float = 0.5

    # Tick size in dollars.
    tick_size: float = 0.01


# Conservative defaults for backtesting.
DEFAULT_FILL_ASSUMPTIONS = FillAssumptions()


def fill_probability(
    taker_volume_at_price: float,
    resting_depth_at_price: float,
    our_size: float,
    assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
) -> float:
    """Estimate the probability that our limit order fills.

    Model: within the fill window, taker volume V arrives at our price level.
    We sit behind (queue_position_frac × resting_depth) contracts. We fill
    fully if V > queue_ahead + our_size.

    This is a simplified binary model — either we fill fully or not at all.
    Partial fills are not modeled (conservative: we only count full fills).

    Args:
        taker_volume_at_price: Historical taker volume at this price level
            within the fill window (contracts).
        resting_depth_at_price: Total resting limit order depth at this
            price level (contracts, excluding our order).
        our_size: Our order size in contracts.
        assumptions: Fill model parameters.

    Returns:
        Probability of fill in [0, 1]. Returns 0 if insufficient taker flow.
    """
    if taker_volume_at_price < assumptions.min_taker_volume:
        return 0.0

    queue_ahead = assumptions.queue_position_frac * resting_depth_at_price
    volume_needed = queue_ahead + our_size

    if taker_volume_at_price <= 0:
        return 0.0

    # Simple ratio model: P(fill) = min(1, taker_volume / volume_needed)
    # Capped at 1.0 — more volume than needed just means higher certainty.
    prob = min(1.0, taker_volume_at_price / volume_needed) if volume_needed > 0 else 1.0
    return prob


def adverse_selection_cost(
    contracts: float,
    post_fill_drift: float | None = None,
    assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
) -> float:
    """Estimate adverse selection cost per fill.

    When a limit order fills, it's often because an informed taker moved the
    price through our level. The post-fill price drift measures how much the
    "true" price moved against us after we were filled.

    Args:
        contracts: Number of contracts filled.
        post_fill_drift: Observed average post-fill price drift (in price
            units, e.g. 0.02 = 2 cents). If None, uses the conservative
            default from assumptions.
        assumptions: Fill model parameters.

    Returns:
        Adverse selection cost in USDC (positive = cost to us).
    """
    if post_fill_drift is not None:
        return abs(post_fill_drift) * contracts

    # Default: assume adverse_selection_ticks × tick_size drift per contract.
    return assumptions.adverse_selection_ticks * assumptions.tick_size * contracts


def expected_fill_cost(
    price: float,
    contracts: float,
    taker_volume_at_price: float,
    resting_depth_at_price: float,
    theta: float = 0.050,
    assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
    post_fill_drift: float | None = None,
) -> dict[str, float]:
    """Compute expected all-in cost for a maker limit order.

    Combines:
      1. Fill probability (determines if we even get into the position)
      2. Maker fee = $0 (makers never pay fees)
      3. Adverse selection cost (post-fill drift)
      4. Exit cost: taker fee on the closing leg (we assume taker exit at
         resolution or via market order)

    This does NOT include the maker rebate (optimistic, pooled daily,
    hard to predict). Omitting it is conservative.

    Args:
        price: Our limit order price in [0.01, 0.99].
        contracts: Order size in contracts.
        taker_volume_at_price: Historical taker volume at this price.
        resting_depth_at_price: Resting depth at this price (excl. us).
        theta: Fee coefficient for market category.
        assumptions: Fill model parameters.
        post_fill_drift: Observed drift if available.

    Returns:
        Dict with:
          - fill_prob: estimated fill probability
          - entry_fee: always 0 (maker)
          - exit_fee: taker fee on the closing trade
          - adverse_selection: expected adverse selection cost
          - total_cost: sum of all costs (conditional on fill)
          - expected_cost: total_cost × fill_prob
    """
    from src.costs.fees import taker_fee as calc_taker_fee

    fp = fill_probability(
        taker_volume_at_price, resting_depth_at_price, contracts, assumptions
    )

    entry_fee = 0.0  # maker
    exit_fee = calc_taker_fee(price, contracts, theta)
    adv_sel = adverse_selection_cost(contracts, post_fill_drift, assumptions)

    total = entry_fee + exit_fee + adv_sel

    return {
        "fill_prob": round(fp, 4),
        "entry_fee": entry_fee,
        "exit_fee": round(exit_fee, 5),
        "adverse_selection": round(adv_sel, 5),
        "total_cost": round(total, 5),
        "expected_cost": round(total * fp, 5),
    }
