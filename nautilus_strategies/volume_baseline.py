# MIT License — pmgetrich project
# This file is our strategy code, kept separate from LGPL NautilusTrader components.

"""
Volume spike baseline strategy for NautilusTrader.

Implements the same rolling volume spike detector used in our statistical
backtest (scripts/backtest_volume_baseline.py). Enters BUY-only when
volume exceeds a rolling percentile threshold, matching the approach
from our TPP comparison study.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal

from strategies.core import LongOnlyPredictionMarketStrategy
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import StrategyConfig


class TradeTickVolumeBaselineConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(100)
    volume_window: int = 60
    volume_percentile: float = 90.0
    cooldown_ticks: int = 50
    take_profit: float = 0.015
    stop_loss: float = 0.02


class TradeTickVolumeBaselineStrategy(LongOnlyPredictionMarketStrategy):
    """
    BUY-only strategy triggered by rolling volume spikes.

    Tracks cumulative trade volume in a sliding window. When volume
    exceeds the Nth percentile of historical windows, and net trade
    direction is positive (more buying), enters a long position.
    """

    def __init__(self, config: TradeTickVolumeBaselineConfig) -> None:
        super().__init__(config)
        self._window_sizes: deque[float] = deque(maxlen=config.volume_window)
        self._window_prices: deque[float] = deque(maxlen=config.volume_window)
        self._historical_volumes: deque[float] = deque(maxlen=1000)

        self._tick_count: int = 0
        self._last_signal_tick: int = 0
        self._total_signals: int = 0

    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        size = float(tick.size)

        self._window_sizes.append(size)
        self._window_prices.append(price)
        self._tick_count += 1

        if self._pending:
            return

        # Risk management for existing positions
        if self._in_position():
            self._risk_exit(
                price=price,
                take_profit=self.config.take_profit,
                stop_loss=self.config.stop_loss,
            )
            return

        # Need full window before generating signals
        if len(self._window_sizes) < self.config.volume_window:
            return

        # Compute rolling volume
        current_volume = sum(self._window_sizes)
        self._historical_volumes.append(current_volume)

        # Need enough history for percentile calculation
        if len(self._historical_volumes) < 50:
            return

        # Cooldown
        if self._tick_count - self._last_signal_tick < self.config.cooldown_ticks:
            return

        # Check if volume exceeds percentile threshold
        sorted_vols = sorted(self._historical_volumes)
        pct_idx = int(len(sorted_vols) * self.config.volume_percentile / 100.0)
        pct_idx = min(pct_idx, len(sorted_vols) - 1)
        threshold = sorted_vols[pct_idx]

        if current_volume <= threshold:
            return

        # Check net direction: more buying than selling
        # Use price changes as a proxy: positive net change = buying pressure
        prices = list(self._window_prices)
        if len(prices) < 2:
            return

        net_change = prices[-1] - prices[0]
        if net_change <= 0:
            return  # BUY-only: skip if net direction is down

        self._total_signals += 1
        self._last_signal_tick = self._tick_count
        self._submit_entry(reference_price=price, visible_size=size)

    def on_reset(self) -> None:
        super().on_reset()
        self._window_sizes.clear()
        self._window_prices.clear()
        self._historical_volumes.clear()
        self._tick_count = 0
        self._last_signal_tick = 0
        self._total_signals = 0
