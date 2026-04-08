"""Walk-forward backtest of whale-tailing strategies on weather markets.

Uses **rolling windows** to avoid survivorship bias: in each period,
wallets are ranked using only lookback data, then tested on the
subsequent forward period.  Windows slide forward and wallets are
re-ranked each period.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import duckdb

from src.store.db import connect

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WhaleProfile:
    """A trader's performance in a lookback window."""

    address: str
    n_markets: int
    total_pnl: float
    total_wagered: float
    roi: float


@dataclass
class Position:
    """A single simulated position from following a whale signal."""

    market_id: str
    question: str
    whale_address: str
    direction: str  # "YES" or "NO"
    entry_price: float
    outcome: float  # 0.0 or 1.0
    pnl: float
    entry_timestamp: int
    period_idx: int  # which forward period this belongs to


@dataclass
class PeriodResult:
    """Results for a single forward period."""

    period_idx: int
    lookback_start: str
    lookback_end: str
    forward_start: str
    forward_end: str
    n_lookback_markets: int
    n_forward_markets: int
    top_whale_addrs: list[str]
    bottom_whale_addrs: list[str]
    positions: list[Position]
    total_pnl: float
    win_rate: float
    n_bets: int


@dataclass
class StrategyResult:
    """Aggregate metrics for one strategy across all periods."""

    name: str
    periods: list[PeriodResult]
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
    """Full rolling-window backtest results."""

    strategies: list[StrategyResult]
    n_periods: int
    window_lookback_days: int
    window_forward_days: int
    total_markets: int
    whale_persistence: float  # rank correlation between adjacent periods


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


def _spearman_rank_corr(a: list[str], b: list[str]) -> float:
    """Spearman rank correlation between two ranked address lists.

    Only considers addresses present in both lists.
    """
    common = set(a) & set(b)
    if len(common) < 3:
        return 0.0
    rank_a = {addr: i for i, addr in enumerate(a) if addr in common}
    rank_b = {addr: i for i, addr in enumerate(b) if addr in common}
    n = len(common)
    d_sq = sum((rank_a[addr] - rank_b[addr]) ** 2 for addr in common)
    return 1.0 - 6.0 * d_sq / (n * (n * n - 1))


def _build_result(
    name: str,
    periods: list[PeriodResult],
    bet_size: float,
) -> StrategyResult:
    all_positions = [p for pr in periods for p in pr.positions]
    if not all_positions:
        return StrategyResult(
            name=name,
            periods=periods,
            positions=[],
            total_pnl=0.0,
            win_rate=0.0,
            roi=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            n_markets=0,
            avg_edge=0.0,
        )

    pnl_list = [p.pnl for p in all_positions]
    total_pnl = sum(pnl_list)
    wins = sum(1 for x in pnl_list if x > 0)
    n_markets = len({p.market_id for p in all_positions})
    total_wagered = len(all_positions) * bet_size

    edges: list[float] = []
    for p in all_positions:
        if p.direction == "YES":
            edges.append(abs(p.entry_price - 0.5))
        else:
            edges.append(abs((1.0 - p.entry_price) - 0.5))

    return StrategyResult(
        name=name,
        periods=periods,
        positions=all_positions,
        total_pnl=total_pnl,
        win_rate=wins / len(all_positions),
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
    """Rolling-window walk-forward whale-tailing backtest.

    For each window:
      1. Rank wallets by PnL in the lookback period (only past data).
      2. In the forward period, simulate tailing (or fading) top/bottom wallets.
      3. Slide window forward and repeat.
    """

    def __init__(
        self, con: duckdb.DuckDBPyConnection, bet_size: float = 100.0
    ) -> None:
        self._con = con
        self._bet_size = bet_size

    @classmethod
    def create(cls, bet_size: float = 100.0) -> WhaleBacktester:
        return cls(connect(), bet_size)

    # -- time windows -------------------------------------------------------

    @staticmethod
    def _generate_windows(
        earliest: datetime,
        latest: datetime,
        lookback_days: int,
        forward_days: int,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """Generate (lb_start, lb_end, fwd_start, fwd_end) windows."""
        windows = []
        # First forward period starts after one full lookback period
        fwd_start = earliest + timedelta(days=lookback_days)
        while fwd_start < latest:
            lb_start = fwd_start - timedelta(days=lookback_days)
            lb_end = fwd_start
            fwd_end = min(fwd_start + timedelta(days=forward_days), latest)
            if fwd_end <= fwd_start:
                break
            windows.append((lb_start, lb_end, fwd_start, fwd_end))
            fwd_start = fwd_end
        return windows

    # -- whale identification per window ------------------------------------

    def _rank_wallets(
        self, lb_start: str, lb_end: str, min_markets: int = 5
    ) -> list[WhaleProfile]:
        """Rank wallets by directional PnL in the lookback window."""
        rows = self._con.execute(
            """
            WITH window_markets AS (
                SELECT id, outcome FROM weather_resolved
                WHERE end_date >= CAST(? AS TIMESTAMPTZ)
                  AND end_date <  CAST(? AS TIMESTAMPTZ)
            ),
            taker_pnl AS (
                SELECT
                    t.taker AS addr, t.market_id, t.usd_amount,
                    CASE
                        WHEN t.side='BUY'  AND wm.outcome=1
                            THEN t.usd_amount * (1.0 / t.price - 1.0)
                        WHEN t.side='BUY'  AND wm.outcome=0
                            THEN -t.usd_amount
                        WHEN t.side='SELL' AND wm.outcome=0
                            THEN t.usd_amount
                        WHEN t.side='SELL' AND wm.outcome=1
                            THEN -t.usd_amount * (1.0 / t.price - 1.0)
                    END AS pnl
                FROM trades t
                JOIN window_markets wm ON t.market_id = wm.id
                WHERE t.price > 0 AND t.price < 1
            ),
            maker_pnl AS (
                SELECT
                    t.maker AS addr, t.market_id, t.usd_amount,
                    CASE
                        WHEN t.side='BUY'  AND wm.outcome=0
                            THEN t.usd_amount
                        WHEN t.side='BUY'  AND wm.outcome=1
                            THEN -t.usd_amount * (1.0 / t.price - 1.0)
                        WHEN t.side='SELL' AND wm.outcome=1
                            THEN t.usd_amount * (1.0 / t.price - 1.0)
                        WHEN t.side='SELL' AND wm.outcome=0
                            THEN -t.usd_amount
                    END AS pnl
                FROM trades t
                JOIN window_markets wm ON t.market_id = wm.id
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
            HAVING count(DISTINCT market_id) >= ?
            ORDER BY total_pnl DESC
            """,
            [lb_start, lb_end, min_markets],
        ).fetchall()

        return [
            WhaleProfile(
                address=r[0],
                n_markets=r[1],
                total_pnl=float(r[2]) if r[2] is not None else 0.0,
                total_wagered=float(r[3]) if r[3] is not None else 0.0,
                roi=float(r[4]) if r[4] is not None else 0.0,
            )
            for r in rows
        ]

    # -- simulation for one forward window ----------------------------------

    def _simulate_window(
        self,
        fwd_start: str,
        fwd_end: str,
        whale_addrs: set[str],
        fade: bool,
        period_idx: int,
    ) -> list[Position]:
        """Simulate tailing/fading in one forward window."""
        if not whale_addrs:
            return []

        mkt_rows = self._con.execute(
            """
            SELECT id, question, outcome
            FROM weather_resolved
            WHERE end_date >= CAST(? AS TIMESTAMPTZ)
              AND end_date <  CAST(? AS TIMESTAMPTZ)
            """,
            [fwd_start, fwd_end],
        ).fetchall()
        mkt_info = {r[0]: (r[1], r[2]) for r in mkt_rows}
        if not mkt_info:
            return []

        addr_list = list(whale_addrs)
        ph = ",".join(["?"] * len(addr_list))

        # Markets where at least one tracked whale was taker
        whale_mkt_rows = self._con.execute(
            f"""
            SELECT DISTINCT t.market_id
            FROM trades t
            JOIN weather_resolved w ON t.market_id = w.id
            WHERE w.end_date >= CAST(? AS TIMESTAMPTZ)
              AND w.end_date <  CAST(? AS TIMESTAMPTZ)
              AND t.taker IN ({ph})
            """,
            [fwd_start, fwd_end] + addr_list,
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
                    if j + 1 >= len(trades):
                        break

                    _, entry_ts, _, _, entry_price = trades[j + 1]
                    if entry_price <= 0 or entry_price >= 1:
                        continue

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
                            period_idx=period_idx,
                        )
                    )
                    entered = True

        positions.sort(key=lambda p: p.entry_timestamp)
        return positions

    # -- random baseline ----------------------------------------------------

    def _simulate_random_window(
        self,
        fwd_start: str,
        fwd_end: str,
        n_bets: int,
        period_idx: int,
        rng: random.Random,
    ) -> list[Position]:
        """Random baseline: bet on random markets at random direction."""
        mkt_rows = self._con.execute(
            """
            SELECT id, question, outcome
            FROM weather_resolved
            WHERE end_date >= CAST(? AS TIMESTAMPTZ)
              AND end_date <  CAST(? AS TIMESTAMPTZ)
            """,
            [fwd_start, fwd_end],
        ).fetchall()
        if not mkt_rows:
            return []

        # Get a representative entry price per market (median trade price)
        mkt_info = {r[0]: (r[1], r[2]) for r in mkt_rows}
        mkt_ids = list(mkt_info.keys())
        if not mkt_ids:
            return []

        id_ph = ",".join(["?"] * len(mkt_ids))
        price_rows = self._con.execute(
            f"""
            SELECT market_id, median(price) AS med_price, max(timestamp) AS ts
            FROM trades
            WHERE market_id IN ({id_ph}) AND price > 0 AND price < 1
            GROUP BY market_id
            """,
            mkt_ids,
        ).fetchall()
        prices = {r[0]: (float(r[1]), int(r[2])) for r in price_rows}

        # Sample random markets (with replacement if n_bets > n_markets)
        available = [mid for mid in mkt_ids if mid in prices]
        if not available:
            return []

        sample = rng.choices(available, k=min(n_bets, len(available)))
        positions: list[Position] = []
        for mid in sample:
            if mid in {p.market_id for p in positions}:
                continue  # one position per market
            question, outcome = mkt_info[mid]
            entry_price, ts = prices[mid]
            direction = rng.choice(["YES", "NO"])

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
                    whale_address="random",
                    direction=direction,
                    entry_price=entry_price,
                    outcome=outcome,
                    pnl=pnl,
                    entry_timestamp=ts,
                    period_idx=period_idx,
                )
            )
        return positions

    # -- public entry point -------------------------------------------------

    def run(
        self,
        lookback_days: int = 90,
        forward_days: int = 30,
        top_n: int = 10,
        bottom_n: int = 5,
        bet_size: float = 100.0,
        random_seed: int = 42,
    ) -> WhaleBacktestSummary:
        """Run the rolling-window walk-forward backtest."""
        self._bet_size = bet_size
        rng = random.Random(random_seed)

        # Time range
        row = self._con.execute("""
            SELECT min(end_date), max(end_date), count(*)
            FROM weather_resolved
        """).fetchone()
        earliest_ts, latest_ts, total_markets = row
        earliest = earliest_ts.replace(tzinfo=None) if hasattr(earliest_ts, 'replace') else datetime.fromisoformat(str(earliest_ts).replace('+00:00', ''))
        latest = latest_ts.replace(tzinfo=None) if hasattr(latest_ts, 'replace') else datetime.fromisoformat(str(latest_ts).replace('+00:00', ''))

        windows = self._generate_windows(
            earliest, latest, lookback_days, forward_days
        )

        print(
            f"Rolling windows: {len(windows)} periods  "
            f"({lookback_days}d lookback, {forward_days}d forward)"
        )
        print(f"Date range: {earliest.date()} → {latest.date()}")
        print(f"Total weather markets: {total_markets}")

        # Strategy accumulators
        strategy_periods: dict[str, list[PeriodResult]] = {
            "tail-top-5": [],
            "tail-top-10": [],
            "fade-bottom-5": [],
            "combined": [],
            "random": [],
        }

        # For persistence tracking: top-N addresses per period
        period_top_rankings: list[list[str]] = []

        for idx, (lb_start, lb_end, fwd_start, fwd_end) in enumerate(windows):
            lb_s = str(lb_start)
            lb_e = str(lb_end)
            fw_s = str(fwd_start)
            fw_e = str(fwd_end)

            # Rank wallets in lookback
            profiles = self._rank_wallets(lb_s, lb_e)
            if not profiles:
                print(f"  Period {idx}: no wallets in lookback, skipping")
                continue

            top_whales = profiles[:top_n]
            bottom_whales = profiles[-bottom_n:]
            top5 = {w.address for w in top_whales[:5]}
            top10 = {w.address for w in top_whales[:10]}
            bot5 = {w.address for w in bottom_whales}

            period_top_rankings.append([w.address for w in top_whales[:top_n]])

            # Count forward markets
            n_fwd = self._con.execute(
                """
                SELECT count(*) FROM weather_resolved
                WHERE end_date >= CAST(? AS TIMESTAMPTZ)
                  AND end_date <  CAST(? AS TIMESTAMPTZ)
                """,
                [fw_s, fw_e],
            ).fetchone()[0]

            n_lb = self._con.execute(
                """
                SELECT count(*) FROM weather_resolved
                WHERE end_date >= CAST(? AS TIMESTAMPTZ)
                  AND end_date <  CAST(? AS TIMESTAMPTZ)
                """,
                [lb_s, lb_e],
            ).fetchone()[0]

            # Simulate each strategy
            for name, addrs, fade in [
                ("tail-top-5", top5, False),
                ("tail-top-10", top10, False),
                ("fade-bottom-5", bot5, True),
            ]:
                positions = self._simulate_window(
                    fw_s, fw_e, addrs, fade, idx
                )
                pnl = sum(p.pnl for p in positions)
                wins = sum(1 for p in positions if p.pnl > 0)
                strategy_periods[name].append(
                    PeriodResult(
                        period_idx=idx,
                        lookback_start=lb_s,
                        lookback_end=lb_e,
                        forward_start=fw_s,
                        forward_end=fw_e,
                        n_lookback_markets=n_lb,
                        n_forward_markets=n_fwd,
                        top_whale_addrs=[w.address for w in top_whales[:5]],
                        bottom_whale_addrs=[w.address for w in bottom_whales],
                        positions=positions,
                        total_pnl=pnl,
                        win_rate=wins / len(positions) if positions else 0.0,
                        n_bets=len(positions),
                    )
                )

            # Combined: tail-top-5 + fade-bottom-5, skip conflicts
            tail_pos = self._simulate_window(fw_s, fw_e, top5, False, idx)
            fade_pos = self._simulate_window(fw_s, fw_e, bot5, True, idx)
            tail_by_mkt = {p.market_id: p for p in tail_pos}
            fade_by_mkt = {p.market_id: p for p in fade_pos}
            combined: list[Position] = []
            for mid in set(tail_by_mkt) | set(fade_by_mkt):
                t = tail_by_mkt.get(mid)
                f = fade_by_mkt.get(mid)
                if t and f:
                    if t.direction == f.direction:
                        combined.append(t)
                elif t:
                    combined.append(t)
                elif f:
                    combined.append(f)
            combined.sort(key=lambda p: p.entry_timestamp)
            pnl = sum(p.pnl for p in combined)
            wins = sum(1 for p in combined if p.pnl > 0)
            strategy_periods["combined"].append(
                PeriodResult(
                    period_idx=idx,
                    lookback_start=lb_s,
                    lookback_end=lb_e,
                    forward_start=fw_s,
                    forward_end=fw_e,
                    n_lookback_markets=n_lb,
                    n_forward_markets=n_fwd,
                    top_whale_addrs=[w.address for w in top_whales[:5]],
                    bottom_whale_addrs=[w.address for w in bottom_whales],
                    positions=combined,
                    total_pnl=pnl,
                    win_rate=wins / len(combined) if combined else 0.0,
                    n_bets=len(combined),
                )
            )

            # Random baseline: match tail-top-5 position count
            n_tail5 = len(strategy_periods["tail-top-5"][-1].positions)
            rand_pos = self._simulate_random_window(
                fw_s, fw_e, max(n_tail5, 10), idx, rng
            )
            rpnl = sum(p.pnl for p in rand_pos)
            rwins = sum(1 for p in rand_pos if p.pnl > 0)
            strategy_periods["random"].append(
                PeriodResult(
                    period_idx=idx,
                    lookback_start=lb_s,
                    lookback_end=lb_e,
                    forward_start=fw_s,
                    forward_end=fw_e,
                    n_lookback_markets=n_lb,
                    n_forward_markets=n_fwd,
                    top_whale_addrs=[],
                    bottom_whale_addrs=[],
                    positions=rand_pos,
                    total_pnl=rpnl,
                    win_rate=rwins / len(rand_pos) if rand_pos else 0.0,
                    n_bets=len(rand_pos),
                )
            )

            print(
                f"  Period {idx}: "
                f"{lb_start.date()} → {fwd_end.date()}  "
                f"fwd_mkts={n_fwd}  "
                f"tail5_bets={n_tail5}  "
                f"tail5_pnl=${strategy_periods['tail-top-5'][-1].total_pnl:+,.0f}"
            )

        # Build aggregate results
        strategies = [
            _build_result(name, periods, bet_size)
            for name, periods in strategy_periods.items()
        ]

        # Whale persistence: avg Spearman rank correlation between adjacent periods
        correlations: list[float] = []
        for i in range(len(period_top_rankings) - 1):
            corr = _spearman_rank_corr(
                period_top_rankings[i], period_top_rankings[i + 1]
            )
            correlations.append(corr)
        persistence = (
            sum(correlations) / len(correlations) if correlations else 0.0
        )

        return WhaleBacktestSummary(
            strategies=strategies,
            n_periods=len(windows),
            window_lookback_days=lookback_days,
            window_forward_days=forward_days,
            total_markets=total_markets,
            whale_persistence=persistence,
        )
