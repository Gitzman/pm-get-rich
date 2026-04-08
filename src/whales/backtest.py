"""Walk-forward backtest of whale-tailing strategies on weather markets.

Identifies top/bottom weather traders in a training period, then simulates
following (or fading) their trades in a held-out test period.  Entry is at
the *next* trade's price after a whale signal to avoid lookahead.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import duckdb

from src.store.db import connect


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WhaleProfile:
    """A trader's performance summary over a period."""

    address: str
    n_markets: int
    total_pnl: float
    total_wagered: float
    roi: float


@dataclass
class Position:
    """A single simulated position opened by following a whale signal."""

    market_id: str
    question: str
    whale_address: str
    direction: str  # "YES" or "NO"
    entry_price: float
    outcome: float  # 0.0 or 1.0
    pnl: float
    entry_timestamp: int


@dataclass
class StrategyResult:
    """Aggregate metrics for one strategy variant."""

    name: str
    positions: list[Position]
    total_pnl: float
    win_rate: float
    roi: float
    sharpe: float
    max_drawdown: float
    n_markets: int
    avg_edge: float


@dataclass
class WhaleBacktestSummary:
    """Full walk-forward backtest results."""

    strategies: list[StrategyResult]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_markets: int
    n_test_markets: int
    train_whales_top: list[WhaleProfile]
    train_whales_bottom: list[WhaleProfile]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _sharpe(pnl_list: list[float]) -> float:
    if len(pnl_list) < 2:
        return 0.0
    mean = sum(pnl_list) / len(pnl_list)
    var = sum((x - mean) ** 2 for x in pnl_list) / (len(pnl_list) - 1)
    std = var**0.5
    return mean / std if std > 0 else 0.0


def _max_drawdown(pnl_list: list[float]) -> float:
    if not pnl_list:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnl_list:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _build_result(
    name: str, positions: list[Position], bet_size: float
) -> StrategyResult:
    if not positions:
        return StrategyResult(
            name=name,
            positions=[],
            total_pnl=0.0,
            win_rate=0.0,
            roi=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            n_markets=0,
            avg_edge=0.0,
        )

    pnl_list = [p.pnl for p in positions]
    total_pnl = sum(pnl_list)
    wins = sum(1 for x in pnl_list if x > 0)
    n_markets = len({p.market_id for p in positions})
    total_wagered = len(positions) * bet_size

    edges: list[float] = []
    for p in positions:
        # Edge = how far our entry price is from 0.5 in our direction
        if p.direction == "YES":
            edges.append(abs(p.entry_price - 0.5))
        else:
            edges.append(abs((1.0 - p.entry_price) - 0.5))

    return StrategyResult(
        name=name,
        positions=positions,
        total_pnl=total_pnl,
        win_rate=wins / len(positions),
        roi=total_pnl / total_wagered if total_wagered > 0 else 0.0,
        sharpe=_sharpe(pnl_list),
        max_drawdown=_max_drawdown(pnl_list),
        n_markets=n_markets,
        avg_edge=sum(edges) / len(edges) if edges else 0.0,
    )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class WhaleBacktester:
    """Walk-forward whale-tailing backtest on weather markets.

    1. Splits resolved weather markets chronologically into train/test.
    2. Identifies top and bottom traders from the training period.
    3. Simulates tailing (or fading) those traders on the test period.
    """

    def __init__(
        self, con: duckdb.DuckDBPyConnection, bet_size: float = 100.0
    ) -> None:
        self._con = con
        self._bet_size = bet_size

    @classmethod
    def create(cls, bet_size: float = 100.0) -> WhaleBacktester:
        return cls(connect(), bet_size)

    # -- train/test split ---------------------------------------------------

    def _split_date(self) -> tuple[str, str, str, str]:
        """Return (train_start, train_end, test_start, test_end) as strings."""
        row = self._con.execute("""
            SELECT
                min(end_date)::VARCHAR,
                max(end_date)::VARCHAR,
                (SELECT end_date::VARCHAR
                 FROM weather_resolved
                 ORDER BY end_date
                 LIMIT 1
                 OFFSET (SELECT count(*) / 2 FROM weather_resolved)
                ) AS median_date
            FROM weather_resolved
        """).fetchone()
        earliest, latest, median = row
        return earliest, median, median, latest

    # -- whale identification -----------------------------------------------

    def _identify_whales(
        self, cutoff: str, top_n: int = 10, bottom_n: int = 5
    ) -> tuple[list[WhaleProfile], list[WhaleProfile]]:
        """Rank traders by directional PnL on train-period weather markets.

        Uses both maker and taker sides (maker direction is flipped).
        """
        rows = self._con.execute(
            """
            WITH train_markets AS (
                SELECT id, outcome FROM weather_resolved
                WHERE end_date <= CAST(? AS TIMESTAMPTZ)
            ),
            -- taker side: side directly gives intent
            taker_pnl AS (
                SELECT
                    t.taker AS addr, t.market_id, t.usd_amount,
                    CASE
                        WHEN t.side='BUY'  AND tm.outcome=1
                            THEN t.usd_amount * (1.0 / t.price - 1.0)
                        WHEN t.side='BUY'  AND tm.outcome=0
                            THEN -t.usd_amount
                        WHEN t.side='SELL' AND tm.outcome=0
                            THEN t.usd_amount
                        WHEN t.side='SELL' AND tm.outcome=1
                            THEN -t.usd_amount * (1.0 / t.price - 1.0)
                    END AS pnl
                FROM trades t
                JOIN train_markets tm ON t.market_id = tm.id
                WHERE t.price > 0 AND t.price < 1
            ),
            -- maker side: direction is opposite of trade side
            maker_pnl AS (
                SELECT
                    t.maker AS addr, t.market_id, t.usd_amount,
                    CASE
                        -- maker of BUY sold YES
                        WHEN t.side='BUY'  AND tm.outcome=0
                            THEN t.usd_amount
                        WHEN t.side='BUY'  AND tm.outcome=1
                            THEN -t.usd_amount * (1.0 / t.price - 1.0)
                        -- maker of SELL bought YES
                        WHEN t.side='SELL' AND tm.outcome=1
                            THEN t.usd_amount * (1.0 / t.price - 1.0)
                        WHEN t.side='SELL' AND tm.outcome=0
                            THEN -t.usd_amount
                    END AS pnl
                FROM trades t
                JOIN train_markets tm ON t.market_id = tm.id
                WHERE t.price > 0 AND t.price < 1
            ),
            all_pnl AS (
                SELECT * FROM taker_pnl
                UNION ALL
                SELECT * FROM maker_pnl
            )
            SELECT
                addr,
                count(DISTINCT market_id) AS n_markets,
                sum(pnl)        AS total_pnl,
                sum(usd_amount) AS total_wagered,
                sum(pnl) / NULLIF(sum(usd_amount), 0) AS roi
            FROM all_pnl
            GROUP BY addr
            HAVING count(DISTINCT market_id) >= 10
            ORDER BY total_pnl DESC
            """,
            [cutoff],
        ).fetchall()

        cols = [d[0] for d in self._con.description]
        profiles = [WhaleProfile(**dict(zip(cols, r))) for r in rows]
        return profiles[:top_n], profiles[-bottom_n:]

    # -- simulation ---------------------------------------------------------

    def _simulate(
        self,
        cutoff: str,
        whale_addrs: set[str],
        fade: bool = False,
    ) -> list[Position]:
        """Simulate tailing or fading *whale_addrs* on test-period markets.

        For each test market with whale taker activity:
          1. Walk trades chronologically.
          2. First taker trade by a tracked whale → record signal.
          3. Enter at the *next* trade's price (no lookahead).
          4. Hold to resolution.
        """
        # Test-period markets
        mkt_rows = self._con.execute(
            """
            SELECT id, question, outcome
            FROM weather_resolved
            WHERE end_date > CAST(? AS TIMESTAMPTZ)
            """,
            [cutoff],
        ).fetchall()
        mkt_info = {r[0]: (r[1], r[2]) for r in mkt_rows}
        if not mkt_info:
            return []

        addr_list = list(whale_addrs)
        ph = ",".join(["?"] * len(addr_list))

        # Markets where at least one whale was taker
        whale_mkt_rows = self._con.execute(
            f"""
            SELECT DISTINCT t.market_id
            FROM trades t
            JOIN weather_resolved w ON t.market_id = w.id
            WHERE w.end_date > CAST(? AS TIMESTAMPTZ)
              AND t.taker IN ({ph})
            """,
            [cutoff] + addr_list,
        ).fetchall()
        whale_mkt_ids = [r[0] for r in whale_mkt_rows]
        if not whale_mkt_ids:
            return []

        positions: list[Position] = []
        batch_size = 500

        for i in range(0, len(whale_mkt_ids), batch_size):
            batch = whale_mkt_ids[i : i + batch_size]
            id_ph = ",".join(["?"] * len(batch))

            rows = self._con.execute(
                f"""
                SELECT market_id, timestamp, taker, side, price
                FROM trades
                WHERE market_id IN ({id_ph})
                  AND price > 0 AND price < 1
                ORDER BY market_id, timestamp, log_index
                """,
                batch,
            ).fetchall()

            by_market: dict[str, list[tuple]] = defaultdict(list)
            for r in rows:
                by_market[r[0]].append(r)

            for mid, trades in by_market.items():
                if mid not in mkt_info:
                    continue
                question, outcome = mkt_info[mid]
                entered = False

                for j, (_, ts, taker, side, price) in enumerate(trades):
                    if entered:
                        break
                    if taker not in whale_addrs:
                        continue
                    # We have a whale taker signal
                    if j + 1 >= len(trades):
                        break  # no next trade to enter at

                    _, entry_ts, _, _, entry_price = trades[j + 1]
                    if entry_price <= 0 or entry_price >= 1:
                        continue

                    # Whale's intent: BUY=bullish, SELL=bearish
                    if fade:
                        direction = "NO" if side == "BUY" else "YES"
                    else:
                        direction = "YES" if side == "BUY" else "NO"

                    if direction == "YES":
                        pnl = (
                            self._bet_size * (1.0 / entry_price - 1.0)
                            if outcome == 1
                            else -self._bet_size
                        )
                    else:
                        pnl = (
                            self._bet_size * (1.0 / (1.0 - entry_price) - 1.0)
                            if outcome == 0
                            else -self._bet_size
                        )

                    positions.append(
                        Position(
                            market_id=mid,
                            question=(question or mid)[:80],
                            whale_address=taker,
                            direction=direction,
                            entry_price=entry_price,
                            outcome=outcome,
                            pnl=pnl,
                            entry_timestamp=entry_ts,
                        )
                    )
                    entered = True

        positions.sort(key=lambda p: p.entry_timestamp)
        return positions

    # -- public entry point -------------------------------------------------

    def run(
        self,
        top_n: int = 10,
        bottom_n: int = 5,
        bet_size: float = 100.0,
    ) -> WhaleBacktestSummary:
        """Run the full walk-forward backtest."""
        self._bet_size = bet_size

        # 1. Train / test split
        train_start, train_end, test_start, test_end = self._split_date()

        n_train = self._con.execute(
            "SELECT count(*) FROM weather_resolved "
            "WHERE end_date <= CAST(? AS TIMESTAMPTZ)",
            [train_end],
        ).fetchone()[0]
        n_test = self._con.execute(
            "SELECT count(*) FROM weather_resolved "
            "WHERE end_date > CAST(? AS TIMESTAMPTZ)",
            [train_end],
        ).fetchone()[0]

        print(f"Train: {train_start} → {train_end}  ({n_train} markets)")
        print(f"Test:  {test_start} → {test_end}  ({n_test} markets)")

        # 2. Identify whales
        print("\nIdentifying whales from training period...")
        top_whales, bottom_whales = self._identify_whales(
            train_end, top_n, bottom_n
        )

        print(f"\nTop {len(top_whales)} whales:")
        for w in top_whales:
            print(
                f"  {w.address[:14]}..  "
                f"{w.n_markets:>5} mkts  "
                f"${w.total_pnl:>12,.0f}  "
                f"ROI {w.roi:>7.1%}"
            )
        print(f"\nBottom {len(bottom_whales)} losers:")
        for w in bottom_whales:
            print(
                f"  {w.address[:14]}..  "
                f"{w.n_markets:>5} mkts  "
                f"${w.total_pnl:>12,.0f}  "
                f"ROI {w.roi:>7.1%}"
            )

        # 3. Strategy variants
        top5 = {w.address for w in top_whales[:5]}
        top10 = {w.address for w in top_whales[:10]}
        bot5 = {w.address for w in bottom_whales}

        strategies: list[StrategyResult] = []

        for name, addrs, fade in [
            ("tail-top-5", top5, False),
            ("tail-top-10", top10, False),
            ("fade-bottom-5", bot5, True),
        ]:
            print(f"\nSimulating {name}...")
            positions = self._simulate(train_end, addrs, fade=fade)
            result = _build_result(name, positions, bet_size)
            strategies.append(result)
            self._print_result(result)

        # Combined: tail-top-5 + fade-bottom-5, skip market conflicts
        print("\nSimulating combined (tail-top-5 + fade-bottom-5)...")
        tail_pos = self._simulate(train_end, top5, fade=False)
        fade_pos = self._simulate(train_end, bot5, fade=True)

        tail_by_mkt = {p.market_id: p for p in tail_pos}
        fade_by_mkt = {p.market_id: p for p in fade_pos}
        combined: list[Position] = []
        all_mkt_ids = set(tail_by_mkt) | set(fade_by_mkt)
        for mid in all_mkt_ids:
            t = tail_by_mkt.get(mid)
            f = fade_by_mkt.get(mid)
            if t and f:
                if t.direction == f.direction:
                    combined.append(t)  # signals agree
                # else: conflict, skip
            elif t:
                combined.append(t)
            elif f:
                combined.append(f)
        combined.sort(key=lambda p: p.entry_timestamp)

        combined_result = _build_result("combined", combined, bet_size)
        strategies.append(combined_result)
        self._print_result(combined_result)

        return WhaleBacktestSummary(
            strategies=strategies,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train_markets=n_train,
            n_test_markets=n_test,
            train_whales_top=top_whales,
            train_whales_bottom=bottom_whales,
        )

    @staticmethod
    def _print_result(r: StrategyResult) -> None:
        print(
            f"  {r.n_markets} markets, {len(r.positions)} positions  "
            f"PnL ${r.total_pnl:+,.0f}  "
            f"Win {r.win_rate:.1%}  "
            f"Sharpe {r.sharpe:.3f}  "
            f"MaxDD ${r.max_drawdown:,.0f}"
        )
