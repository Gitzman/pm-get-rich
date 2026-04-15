"""Unit tests for the L2 order-book-aware fill simulator.

Tests use synthetic book data to verify queue position tracking, fill
detection, edge cases, and interface compatibility with the statistical
fill model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.costs.book_fills import FillResult, L2FillSimulator, simulate_fill_series
from src.costs.fills import FillAssumptions
from src.ingest.pmxt_loader import BookLevel, BookSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MKT = "test-market-001"
T0 = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)


def _snap(
    t: datetime,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    market_id: str = MKT,
) -> BookSnapshot:
    """Build a BookSnapshot from (price, size) tuples."""
    bid_levels = [BookLevel(p, s) for p, s in bids]
    ask_levels = [BookLevel(p, s) for p, s in asks]
    return BookSnapshot(
        timestamp=t,
        market_id=market_id,
        best_bid=bids[0][0] if bids else 0.0,
        best_ask=asks[0][0] if asks else 1.0,
        bids=sorted(bid_levels, key=lambda l: l.price, reverse=True),
        asks=sorted(ask_levels, key=lambda l: l.price),
    )


def _assumptions(window_s: float = 300.0) -> FillAssumptions:
    return FillAssumptions(fill_window_s=window_s)


# ---------------------------------------------------------------------------
# BookSnapshot unit tests
# ---------------------------------------------------------------------------


class TestBookSnapshot:
    def test_depth_at_price(self):
        snap = _snap(T0, [(0.55, 100), (0.54, 50)], [(0.56, 80), (0.57, 60)])
        assert snap.depth_at_price(0.55, "bid") == 100
        assert snap.depth_at_price(0.56, "ask") == 80
        assert snap.depth_at_price(0.50, "bid") == 0.0  # no level

    def test_depth_at_or_better_bids(self):
        snap = _snap(T0, [(0.55, 100), (0.54, 50), (0.53, 30)], [(0.56, 80)])
        # At or better for bids: prices >= 0.54
        assert snap.depth_at_or_better(0.54, "bid") == 100 + 50  # 0.55 + 0.54

    def test_depth_at_or_better_asks(self):
        snap = _snap(T0, [(0.55, 100)], [(0.56, 80), (0.57, 60), (0.58, 40)])
        # At or better for asks: prices <= 0.57
        assert snap.depth_at_or_better(0.57, "ask") == 80 + 60  # 0.56 + 0.57

    def test_mid_and_spread(self):
        snap = _snap(T0, [(0.55, 100)], [(0.57, 80)])
        assert snap.mid == pytest.approx(0.56)
        assert snap.spread == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Immediate fill (marketable order)
# ---------------------------------------------------------------------------


class TestImmediateFill:
    """Orders that cross the spread should fill immediately."""

    def test_buy_at_ask_fills_immediately(self):
        snaps = [_snap(T0, [(0.55, 100)], [(0.56, 80)])]
        sim = L2FillSimulator(snaps)
        # Buy at 0.56 = at the ask → immediate fill
        result = sim.simulate_order(MKT, 0.56, 10, "buy", T0, _assumptions())
        assert result.filled is True
        assert result.time_to_fill_s == 0.0

    def test_buy_above_ask_fills_immediately(self):
        snaps = [_snap(T0, [(0.55, 100)], [(0.56, 80)])]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.60, 10, "buy", T0, _assumptions())
        assert result.filled is True
        assert result.time_to_fill_s == 0.0

    def test_sell_at_bid_fills_immediately(self):
        snaps = [_snap(T0, [(0.55, 100)], [(0.56, 80)])]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "sell", T0, _assumptions())
        assert result.filled is True
        assert result.time_to_fill_s == 0.0


# ---------------------------------------------------------------------------
# Queue position tracking
# ---------------------------------------------------------------------------


class TestQueueTracking:
    """Orders that join the queue and track position as book evolves."""

    def test_fill_when_taker_flow_clears_queue(self):
        """Place buy at 0.55 with 100 ahead. Taker flow consumes all 100."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 0)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        assert result.filled is True
        assert result.queue_position_initial == 100
        assert result.time_to_fill_s == 60.0

    def test_partial_queue_advancement_no_fill(self):
        """Place buy at 0.55 with 100 ahead. Only 40 consumed → not filled."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 60)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        assert result.filled is False
        assert result.queue_position_final == 60  # 100 - 40

    def test_gradual_fill_across_multiple_snapshots(self):
        """Queue drains gradually over 3 snapshots."""
        snaps = [
            _snap(T0, [(0.55, 90)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=30), [(0.55, 60)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 30)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=90), [(0.55, 0)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        assert result.filled is True
        assert result.queue_position_initial == 90
        assert result.time_to_fill_s == 90.0

    def test_depth_increase_does_not_advance_queue(self):
        """New orders joining behind us don't affect our queue position."""
        snaps = [
            _snap(T0, [(0.55, 50)], [(0.56, 80)]),
            # Depth increased from 50 to 150 → 100 new orders joined behind
            _snap(T0 + timedelta(seconds=60), [(0.55, 150)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        assert result.filled is False
        assert result.queue_position_final == 50  # unchanged

    def test_sell_order_queue_tracking(self):
        """Sell order joins the ask side and tracks queue."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.57, 200)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 100)], [(0.57, 0)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.57, 50, "sell", T0, _assumptions())
        assert result.filled is True
        assert result.queue_position_initial == 200


# ---------------------------------------------------------------------------
# Market crossing (book moves through our price)
# ---------------------------------------------------------------------------


class TestMarketCrossing:
    """Book moves through our limit price → guaranteed fill."""

    def test_buy_filled_when_ask_drops_below(self):
        """Ask drops below our bid → market crossed our price."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.57, 80)]),
            _snap(T0 + timedelta(seconds=30), [(0.55, 100)], [(0.54, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        # Buy at 0.56 — inside the spread initially
        result = sim.simulate_order(MKT, 0.56, 10, "buy", T0, _assumptions())
        assert result.filled is True
        assert result.time_to_fill_s == 30.0

    def test_sell_filled_when_bid_rises_above(self):
        """Bid rises above our ask → market crossed our price."""
        snaps = [
            _snap(T0, [(0.53, 100)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=45), [(0.58, 100)], [(0.59, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        # Sell at 0.57
        result = sim.simulate_order(MKT, 0.57, 10, "sell", T0, _assumptions())
        assert result.filled is True
        assert result.time_to_fill_s == 45.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_snapshots_for_market(self):
        """No book data for the requested market → no fill."""
        snaps = [_snap(T0, [(0.55, 100)], [(0.56, 80)], market_id="other-mkt")]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        assert result.filled is False
        assert result.snapshots_observed == 0

    def test_zero_depth_at_price(self):
        """No resting depth at our price → queue ahead is 0, first taker fills us."""
        snaps = [
            # No depth at 0.54 — we'd be the only order there
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            # Ask drops to 0.54 → crosses our buy at 0.54
            _snap(T0 + timedelta(seconds=30), [(0.53, 100)], [(0.54, 50)]),
        ]
        sim = L2FillSimulator(snaps)
        # Buy at 0.54 — below the best bid, no queue
        result = sim.simulate_order(MKT, 0.54, 10, "buy", T0, _assumptions())
        # The buy at 0.54 is below the best bid of 0.55, so it's a valid limit order
        # queue_ahead = 0 (no depth at 0.54)
        # In the first snapshot T0, there's zero depth at 0.54, so queue_ahead=0.
        # No depth consumed (prev_depth was 0, current is 0) → but it checks
        # market crossing first: ask doesn't drop to 0.54 until snap 2.
        # At snap 2: ask = 0.54 <= price 0.54 → crossed → filled
        assert result.filled is True

    def test_fill_window_expiry(self):
        """Order expires if no fill within the window."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            # Taker flow arrives, but AFTER the 60s window
            _snap(T0 + timedelta(seconds=120), [(0.55, 0)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(
            MKT, 0.55, 10, "buy", T0, _assumptions(window_s=60.0)
        )
        assert result.filled is False

    def test_book_moves_away(self):
        """Book moves away from our price → no fill, no crash."""
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            # Book moves up, away from our bid at 0.50
            _snap(T0 + timedelta(seconds=60), [(0.60, 100)], [(0.62, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.50, 10, "buy", T0, _assumptions())
        assert result.filled is False

    def test_empty_book(self):
        """Empty book (no bids or asks) → no crash."""
        snaps = [_snap(T0, [], [])]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 10, "buy", T0, _assumptions())
        # best_ask = 1.0 (default), price 0.55 < 1.0 → not marketable
        # No depth at 0.55 → queue_ahead = 0
        # No subsequent snapshots → no fill
        assert result.filled is False

    def test_single_contract_order(self):
        """Minimum order size of 1 contract."""
        snaps = [
            _snap(T0, [(0.55, 5)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=30), [(0.55, 0)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)
        result = sim.simulate_order(MKT, 0.55, 1, "buy", T0, _assumptions())
        assert result.filled is True


# ---------------------------------------------------------------------------
# Drop-in interface compatibility
# ---------------------------------------------------------------------------


class TestDropInInterface:
    """Verify the simulator produces output compatible with fills.py."""

    def _make_sim(self) -> L2FillSimulator:
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 0)], [(0.56, 80)]),
        ]
        return L2FillSimulator(snaps)

    def test_fill_probability_returns_float(self):
        sim = self._make_sim()
        prob = sim.fill_probability(MKT, 0.55, 10, "buy", T0)
        assert isinstance(prob, float)
        assert prob in (0.0, 1.0)

    def test_expected_fill_cost_returns_dict(self):
        sim = self._make_sim()
        cost = sim.expected_fill_cost(MKT, 0.55, 10, "buy", T0)
        # Must have all keys from the statistical model
        for key in ("fill_prob", "entry_fee", "exit_fee",
                     "adverse_selection", "total_cost", "expected_cost"):
            assert key in cost, f"Missing key: {key}"

    def test_expected_fill_cost_entry_fee_is_zero(self):
        """Maker orders always have zero entry fee."""
        sim = self._make_sim()
        cost = sim.expected_fill_cost(MKT, 0.55, 10, "buy", T0)
        assert cost["entry_fee"] == 0.0

    def test_expected_fill_cost_unfilled(self):
        """Unfilled order → expected_cost = 0."""
        snaps = [_snap(T0, [(0.55, 1000)], [(0.56, 80)])]
        sim = L2FillSimulator(snaps)
        cost = sim.expected_fill_cost(MKT, 0.55, 10, "buy", T0)
        assert cost["fill_prob"] == 0.0
        assert cost["expected_cost"] == 0.0

    def test_l2_specific_fields_present(self):
        """L2 simulator adds extra diagnostic fields."""
        sim = self._make_sim()
        cost = sim.expected_fill_cost(MKT, 0.55, 10, "buy", T0)
        assert "time_to_fill_s" in cost
        assert "queue_position_initial" in cost
        assert "snapshots_observed" in cost


# ---------------------------------------------------------------------------
# simulate_fill_series
# ---------------------------------------------------------------------------


class TestSimulateFillSeries:
    def test_batch_simulation(self):
        snaps = [
            _snap(T0, [(0.55, 50)], [(0.56, 80)]),
            _snap(T0 + timedelta(seconds=60), [(0.55, 0)], [(0.56, 80)]),
        ]
        sim = L2FillSimulator(snaps)

        orders = [
            {"price": 0.55, "contracts": 10, "side": "buy", "place_time": T0},
            {"price": 0.56, "contracts": 5, "side": "buy", "place_time": T0},
        ]
        results = simulate_fill_series(sim, MKT, orders)
        assert len(results) == 2
        # First order: joins queue at 0.55, filled after queue drains
        assert results[0]["fill_prob"] == 1.0
        # Second order: at the ask → immediate fill
        assert results[1]["fill_prob"] == 1.0


# ---------------------------------------------------------------------------
# Multiple markets
# ---------------------------------------------------------------------------


class TestMultipleMarkets:
    def test_independent_market_simulation(self):
        """Two markets in the same simulator don't interfere."""
        mkt_a = "market-A"
        mkt_b = "market-B"
        snaps = [
            _snap(T0, [(0.55, 100)], [(0.56, 80)], market_id=mkt_a),
            _snap(T0, [(0.40, 50)], [(0.42, 30)], market_id=mkt_b),
            # Market A fills, market B doesn't
            _snap(T0 + timedelta(seconds=60), [(0.55, 0)], [(0.56, 80)], market_id=mkt_a),
            _snap(T0 + timedelta(seconds=60), [(0.40, 50)], [(0.42, 30)], market_id=mkt_b),
        ]
        sim = L2FillSimulator(snaps)

        result_a = sim.simulate_order(mkt_a, 0.55, 10, "buy", T0, _assumptions())
        result_b = sim.simulate_order(mkt_b, 0.40, 10, "buy", T0, _assumptions())

        assert result_a.filled is True
        assert result_b.filled is False  # depth unchanged


# ---------------------------------------------------------------------------
# PMXT loader unit tests (parse_book_snapshot)
# ---------------------------------------------------------------------------


class TestParseBookSnapshot:
    def test_parse_valid_snapshot(self):
        from src.ingest.pmxt_loader import parse_book_snapshot

        row = {
            "market_id": MKT,
            "update_type": "book_snapshot",
            "data": '{"best_bid": 0.55, "best_ask": 0.56, '
                    '"bids": [[0.55, 100], [0.54, 50]], '
                    '"asks": [[0.56, 80], [0.57, 60]], '
                    '"timestamp": 1711360800000}',
        }
        snap = parse_book_snapshot(row)
        assert snap is not None
        assert snap.market_id == MKT
        assert snap.best_bid == 0.55
        assert snap.best_ask == 0.56
        assert len(snap.bids) == 2
        assert len(snap.asks) == 2
        assert snap.bids[0].price == 0.55  # sorted desc
        assert snap.asks[0].price == 0.56  # sorted asc

    def test_parse_price_change_returns_none(self):
        from src.ingest.pmxt_loader import parse_book_snapshot

        row = {
            "market_id": MKT,
            "update_type": "price_change",
            "data": '{"price": 0.55}',
        }
        assert parse_book_snapshot(row) is None

    def test_parse_dict_data(self):
        """Data can be a dict (already parsed) not just a string."""
        from src.ingest.pmxt_loader import parse_book_snapshot

        row = {
            "market_id": MKT,
            "update_type": "book_snapshot",
            "data": {
                "best_bid": 0.50,
                "best_ask": 0.52,
                "bids": [[0.50, 200]],
                "asks": [[0.52, 150]],
                "timestamp": 1711360800000,
            },
        }
        snap = parse_book_snapshot(row)
        assert snap is not None
        assert snap.best_bid == 0.50
