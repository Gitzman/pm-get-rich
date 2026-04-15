"""L2 order-book-aware fill simulator for Polymarket.

Drop-in replacement for the statistical fill model in fills.py.
Instead of estimating fill probability from aggregate taker volume
and resting depth statistics, this module replays actual L2 order book
snapshots to simulate queue position tracking and fill execution.

Model:
  1. At order placement time, reconstruct the book from the most recent
     snapshot. Our order joins the BACK of the queue at our price level.
  2. As subsequent snapshots arrive, track how the queue evolves:
     - Size decreases at our price level → taker flow consumed orders
       ahead of us, advancing our queue position.
     - Size increases → new orders joined behind us (no effect on position).
     - If our queue position reaches zero and more taker flow arrives,
       we are filled.
  3. For exits: same logic but on the opposite side of the book.

Assumptions documented in book_fill_model.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.costs.fees import taker_fee as calc_taker_fee
from src.costs.fills import FillAssumptions, DEFAULT_FILL_ASSUMPTIONS
from src.ingest.pmxt_loader import BookSnapshot


@dataclass(frozen=True)
class FillResult:
    """Result of simulating a single order against L2 book data."""

    filled: bool
    fill_price: float | None = None
    fill_time: datetime | None = None
    queue_position_initial: float = 0.0
    queue_position_final: float = 0.0
    time_to_fill_s: float | None = None
    snapshots_observed: int = 0
    partial_fill_frac: float = 0.0  # fraction of order that could have filled


@dataclass
class L2FillSimulator:
    """Order-book-aware fill simulator using PMXT L2 snapshots.

    Usage:
        snapshots = load_book_snapshots(start, end, condition_ids)
        sim = L2FillSimulator(snapshots)

        # Simulate a single order
        result = sim.simulate_order(
            price=0.55,
            size=100.0,
            side="buy",
            place_time=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
        )

        # Drop-in replacement interface
        prob = sim.fill_probability(price=0.55, size=100.0, side="buy",
                                     place_time=..., assumptions=...)
        cost = sim.expected_fill_cost(price=0.55, contracts=100.0, side="buy",
                                       place_time=..., theta=0.050)
    """

    snapshots: list[BookSnapshot]
    # Index by market_id for fast lookup
    _by_market: dict[str, list[BookSnapshot]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._by_market = {}
        for snap in self.snapshots:
            self._by_market.setdefault(snap.market_id, []).append(snap)
        # Ensure each market's snapshots are sorted by time
        for mkt_snaps in self._by_market.values():
            mkt_snaps.sort(key=lambda s: s.timestamp)

    @property
    def market_ids(self) -> list[str]:
        return list(self._by_market.keys())

    def _get_market_snapshots(
        self, market_id: str, after: datetime
    ) -> list[BookSnapshot]:
        """Get snapshots for a market starting from a given time."""
        snaps = self._by_market.get(market_id, [])
        # Binary search for first snapshot >= after
        lo, hi = 0, len(snaps)
        while lo < hi:
            mid = (lo + hi) // 2
            if snaps[mid].timestamp < after:
                lo = mid + 1
            else:
                hi = mid
        return snaps[lo:]

    def _get_snapshot_at(
        self, market_id: str, at: datetime
    ) -> BookSnapshot | None:
        """Get the most recent snapshot at or before the given time."""
        snaps = self._by_market.get(market_id, [])
        if not snaps:
            return None
        # Binary search for last snapshot <= at
        lo, hi = 0, len(snaps) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if snaps[mid].timestamp <= at:
                result = snaps[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def simulate_order(
        self,
        market_id: str,
        price: float,
        size: float,
        side: str,
        place_time: datetime,
        assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
    ) -> FillResult:
        """Simulate placing a limit order and track it through the book.

        Args:
            market_id: Polymarket condition_id.
            price: Limit price for our order.
            size: Order size in contracts.
            side: "buy" (we place a bid) or "sell" (we place an ask).
            place_time: When we place the order (UTC datetime).
            assumptions: Fill model parameters (uses fill_window_s).

        Returns:
            FillResult with fill status, timing, and queue position info.
        """
        from datetime import timedelta

        deadline = place_time + timedelta(seconds=assumptions.fill_window_s)

        # Determine which side of the book we join
        # Buy order → joins bids (we want someone to sell into us)
        # Sell order → joins asks (we want someone to buy into us)
        book_side = "bid" if side == "buy" else "ask"

        # Get the book state at placement time
        initial_snap = self._get_snapshot_at(market_id, place_time)
        if initial_snap is None:
            return FillResult(
                filled=False,
                queue_position_initial=0.0,
                queue_position_final=0.0,
                snapshots_observed=0,
            )

        # Check if our order is marketable (crosses the spread)
        # Buy at or above best ask → immediate fill (taker, not maker)
        # Sell at or below best bid → immediate fill (taker, not maker)
        if side == "buy" and price >= initial_snap.best_ask:
            return FillResult(
                filled=True,
                fill_price=price,
                fill_time=place_time,
                queue_position_initial=0.0,
                queue_position_final=0.0,
                time_to_fill_s=0.0,
                snapshots_observed=1,
                partial_fill_frac=1.0,
            )
        if side == "sell" and price <= initial_snap.best_bid:
            return FillResult(
                filled=True,
                fill_price=price,
                fill_time=place_time,
                queue_position_initial=0.0,
                queue_position_final=0.0,
                time_to_fill_s=0.0,
                snapshots_observed=1,
                partial_fill_frac=1.0,
            )

        # Calculate initial queue position: total size at our price level
        # that was placed BEFORE us (we're at the back of the queue).
        # Our order is virtual — not reflected in the real book data.
        initial_depth = initial_snap.depth_at_price(price, book_side)
        queue_ahead = initial_depth  # all existing depth is ahead of us

        # Track queue position through subsequent snapshots.
        # We only observe net depth changes between snapshots (interval data).
        # Conservative: attribute net decreases to taker flow, ignore masked
        # flow that was offset by new limit orders arriving between snapshots.
        future_snaps = self._get_market_snapshots(market_id, place_time)
        prev_depth = initial_depth  # depth at our price in previous snapshot
        snapshots_seen = 0

        for snap in future_snaps:
            if snap.timestamp > deadline:
                break
            snapshots_seen += 1

            # Check if the market crossed through our price level.
            # This means aggressive flow swept past our price → definite fill.
            if side == "buy" and snap.best_ask <= price:
                return FillResult(
                    filled=True,
                    fill_price=price,
                    fill_time=snap.timestamp,
                    queue_position_initial=initial_depth,
                    queue_position_final=0.0,
                    time_to_fill_s=(snap.timestamp - place_time).total_seconds(),
                    snapshots_observed=snapshots_seen,
                    partial_fill_frac=1.0,
                )
            if side == "sell" and snap.best_bid >= price:
                return FillResult(
                    filled=True,
                    fill_price=price,
                    fill_time=snap.timestamp,
                    queue_position_initial=initial_depth,
                    queue_position_final=0.0,
                    time_to_fill_s=(snap.timestamp - place_time).total_seconds(),
                    snapshots_observed=snapshots_seen,
                    partial_fill_frac=1.0,
                )

            current_depth = snap.depth_at_price(price, book_side)

            # Net depth decrease at our price → taker flow consumed from front
            depth_consumed = max(0.0, prev_depth - current_depth)

            if depth_consumed > 0:
                queue_ahead = max(0.0, queue_ahead - depth_consumed)

                if queue_ahead <= 0:
                    # All orders ahead of us were consumed, and excess taker
                    # flow would have reached our virtual order → FILLED
                    return FillResult(
                        filled=True,
                        fill_price=price,
                        fill_time=snap.timestamp,
                        queue_position_initial=initial_depth,
                        queue_position_final=0.0,
                        time_to_fill_s=(snap.timestamp - place_time).total_seconds(),
                        snapshots_observed=snapshots_seen,
                        partial_fill_frac=1.0,
                    )

            # Depth increased → new orders joined behind us, no queue advancement
            # (we only track queue_ahead, not total depth)

            prev_depth = current_depth

        # Order expired without fill within the holding window
        return FillResult(
            filled=False,
            fill_price=None,
            fill_time=None,
            queue_position_initial=initial_depth,
            queue_position_final=queue_ahead,
            time_to_fill_s=None,
            snapshots_observed=snapshots_seen,
            partial_fill_frac=0.0,
        )

    def fill_probability(
        self,
        market_id: str,
        price: float,
        size: float,
        side: str,
        place_time: datetime,
        assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
    ) -> float:
        """Estimate fill probability by replaying L2 book data.

        Drop-in replacement for fills.fill_probability(). Instead of using
        aggregate taker volume statistics, replays actual book snapshots.

        Returns:
            Probability of fill in [0, 1]. Binary: 1.0 if filled, 0.0 if not.
            For Monte Carlo estimation over multiple time windows, call
            simulate_order() directly.
        """
        result = self.simulate_order(
            market_id=market_id,
            price=price,
            size=size,
            side=side,
            place_time=place_time,
            assumptions=assumptions,
        )
        return 1.0 if result.filled else 0.0

    def expected_fill_cost(
        self,
        market_id: str,
        price: float,
        contracts: float,
        side: str,
        place_time: datetime,
        theta: float = 0.050,
        assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
    ) -> dict[str, float]:
        """Compute expected all-in cost using L2 book simulation.

        Drop-in replacement for fills.expected_fill_cost(). Uses actual book
        data instead of statistical estimates.

        Returns:
            Dict with same keys as fills.expected_fill_cost():
              - fill_prob, entry_fee, exit_fee, adverse_selection,
                total_cost, expected_cost
            Plus additional L2-specific fields:
              - time_to_fill_s, queue_position_initial, snapshots_observed
        """
        from src.costs.fills import adverse_selection_cost

        result = self.simulate_order(
            market_id=market_id,
            price=price,
            size=contracts,
            side=side,
            place_time=place_time,
            assumptions=assumptions,
        )

        fp = 1.0 if result.filled else 0.0
        entry_fee = 0.0  # maker
        exit_fee = calc_taker_fee(price, contracts, theta)
        adv_sel = adverse_selection_cost(contracts, assumptions=assumptions)

        total = entry_fee + exit_fee + adv_sel

        return {
            "fill_prob": fp,
            "entry_fee": entry_fee,
            "exit_fee": round(exit_fee, 5),
            "adverse_selection": round(adv_sel, 5),
            "total_cost": round(total, 5),
            "expected_cost": round(total * fp, 5),
            # L2-specific fields
            "time_to_fill_s": result.time_to_fill_s,
            "queue_position_initial": result.queue_position_initial,
            "snapshots_observed": result.snapshots_observed,
        }


def simulate_fill_series(
    simulator: L2FillSimulator,
    market_id: str,
    orders: list[dict],
    assumptions: FillAssumptions = DEFAULT_FILL_ASSUMPTIONS,
    theta: float = 0.050,
) -> list[dict[str, float]]:
    """Simulate a series of orders and return cost results.

    Convenience function for backtesting: given a list of order dicts,
    run each through the L2 simulator and return results.

    Args:
        simulator: L2FillSimulator with loaded book data.
        market_id: The market to simulate in.
        orders: List of dicts, each with:
            - price: float
            - contracts: float
            - side: "buy" or "sell"
            - place_time: datetime
        assumptions: Fill model parameters.
        theta: Fee coefficient.

    Returns:
        List of cost result dicts (same as expected_fill_cost output).
    """
    results = []
    for order in orders:
        cost = simulator.expected_fill_cost(
            market_id=market_id,
            price=order["price"],
            contracts=order["contracts"],
            side=order["side"],
            place_time=order["place_time"],
            theta=theta,
            assumptions=assumptions,
        )
        results.append(cost)
    return results
