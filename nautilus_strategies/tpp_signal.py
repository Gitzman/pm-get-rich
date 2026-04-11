# MIT License — pmgetrich project
# This file is our strategy code, kept separate from LGPL NautilusTrader components.

"""
TPP (Temporal Point Process) signal strategy for NautilusTrader.

Wraps the CrossMarketTPP ONNX model to generate BUY-only signals from
trade tick sequences on Polymarket weather markets.

The model was trained on trade event sequences (wallet, city, side, price,
bucket_pos, time_delta, hours_to_res, n_buckets). In the NautilusTrader
context we only observe trade ticks (price, size, timestamp). We map these
to the model's input space with conservative defaults for unobservable
features (unknown wallet, inferred side from price movement).
"""

from __future__ import annotations

import json
import math
from collections import deque
from decimal import Decimal
from pathlib import Path

import numpy as np

from strategies.core import LongOnlyPredictionMarketStrategy
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import StrategyConfig


class TradeTickTPPSignalConfig(StrategyConfig, frozen=True):  # type: ignore[call-arg]
    instrument_id: InstrumentId
    trade_size: Decimal = Decimal(100)
    model_path: str = ""
    vocab_path: str = ""
    city_name: str = ""
    n_buckets: int = 11
    hours_to_resolution: float = 24.0
    context_length: int = 128
    confidence_threshold: float = 0.55
    cooldown_ticks: int = 50
    take_profit: float = 0.015
    stop_loss: float = 0.02


class TradeTickTPPSignalStrategy(LongOnlyPredictionMarketStrategy):
    """
    BUY-only strategy driven by CrossMarketTPP ONNX model inference.

    Accumulates trade tick history, periodically runs the model, and enters
    a long position when the predicted bucket position indicates upward
    movement with sufficient confidence. Only BUY signals are acted on
    (SELL signals are adversely selected per our backtest findings).
    """

    def __init__(self, config: TradeTickTPPSignalConfig) -> None:
        super().__init__(config)

        self._session = None  # lazy init in on_start
        self._vocab: dict | None = None
        self._city_idx: int = 0
        self._n_wallets: int = 501

        # Event history buffer
        self._prices: deque[float] = deque(maxlen=config.context_length)
        self._sizes: deque[float] = deque(maxlen=config.context_length)
        self._timestamps_ms: deque[int] = deque(maxlen=config.context_length)
        self._sides: deque[int] = deque(maxlen=config.context_length)  # 0=BUY, 1=SELL

        self._tick_count: int = 0
        self._last_signal_tick: int = 0
        self._total_signals: int = 0
        self._buy_signals: int = 0

    def _subscribe(self) -> None:
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_start(self) -> None:
        super().on_start()
        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model and vocab metadata."""
        import onnxruntime as ort

        model_path = self.config.model_path
        if not model_path or not Path(model_path).exists():
            self.log.warning(f"Model not found at {model_path} — running in passthrough mode")
            return

        self._session = ort.InferenceSession(model_path)

        vocab_path = self.config.vocab_path
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path) as f:
                self._vocab = json.load(f)
            self._n_wallets = self._vocab.get("n_wallets", 501)
            cities = self._vocab.get("cities", [])
            city = self.config.city_name
            self._city_idx = cities.index(city) if city in cities else 0
        self.log.info(
            f"TPP model loaded: city={self.config.city_name} idx={self._city_idx} "
            f"n_wallets={self._n_wallets}"
        )

    def on_trade_tick(self, tick: TradeTick) -> None:
        price = float(tick.price)
        size = float(tick.size)
        ts_ms = tick.ts_event // 1_000_000  # nanoseconds to milliseconds

        # Infer side from price movement (crude but best we can do from ticks)
        if len(self._prices) > 0:
            side = 0 if price >= self._prices[-1] else 1  # BUY if price up/flat
        else:
            side = 0

        self._prices.append(price)
        self._sizes.append(size)
        self._timestamps_ms.append(ts_ms)
        self._sides.append(side)
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

        # Need enough context before inference
        if len(self._prices) < min(32, self.config.context_length):
            return

        # Cooldown between signals
        if self._tick_count - self._last_signal_tick < self.config.cooldown_ticks:
            return

        # Run inference
        signal = self._compute_signal()
        if signal is not None and signal > self.config.confidence_threshold:
            self._total_signals += 1
            self._buy_signals += 1
            self._last_signal_tick = self._tick_count
            self._submit_entry(reference_price=price, visible_size=size)

    def _compute_signal(self) -> float | None:
        """Run TPP model inference on accumulated tick history.

        Returns predicted bucket position (sigmoid output) or None if model
        unavailable. Values > 0.5 indicate upward price movement.
        """
        if self._session is None:
            return None

        n = len(self._prices)
        prices = list(self._prices)
        timestamps = list(self._timestamps_ms)
        sides = list(self._sides)

        # Build model inputs
        # wallet_idx: unknown wallet (index 0 = unknown token)
        wallet_idx = np.zeros((1, n), dtype=np.int64)

        # city_idx: constant for this market
        city_idx = np.full((1, n), self._city_idx, dtype=np.int64)

        # side_idx
        side_idx = np.array(sides, dtype=np.int64).reshape(1, n)

        # bucket_pos: normalize price to [0, 1] range within observed range
        p_arr = np.array(prices, dtype=np.float32)
        p_min, p_max = p_arr.min(), p_arr.max()
        if p_max > p_min:
            bucket_pos = ((p_arr - p_min) / (p_max - p_min)).reshape(1, n)
        else:
            bucket_pos = np.full((1, n), 0.5, dtype=np.float32)

        # price: raw prices (already in [0, 1] for prediction markets)
        price_input = p_arr.reshape(1, n)

        # time_delta: hours since previous event
        ts_arr = np.array(timestamps, dtype=np.float64)
        dt_ms = np.diff(ts_arr, prepend=ts_arr[0])
        time_delta = (dt_ms / 3_600_000.0).astype(np.float32).reshape(1, n)

        # hours_to_res: decreasing as we approach resolution
        h2r = self.config.hours_to_resolution
        elapsed_hours = (ts_arr - ts_arr[0]) / 3_600_000.0
        hours_to_res = np.maximum(0.0, h2r - elapsed_hours).astype(np.float32).reshape(1, n)

        # n_buckets
        n_buckets = np.full((1, n), float(self.config.n_buckets), dtype=np.float32)

        inputs = {
            "wallet_idx": wallet_idx,
            "city_idx": city_idx,
            "side_idx": side_idx,
            "bucket_pos": bucket_pos,
            "price": price_input,
            "time_delta": time_delta,
            "hours_to_res": hours_to_res,
            "n_buckets": n_buckets,
        }

        try:
            outputs = self._session.run(None, inputs)
            # outputs: [wallet_logits, bucket_pred, time_mu, time_log_sigma]
            bucket_pred = outputs[1]  # shape (1, n)
            # Sigmoid to get probability
            last_pred = float(bucket_pred[0, -1])
            pred_prob = 1.0 / (1.0 + math.exp(-last_pred))
            return pred_prob
        except Exception as e:
            self.log.warning(f"TPP inference failed: {e}")
            return None

    def on_reset(self) -> None:
        super().on_reset()
        self._prices.clear()
        self._sizes.clear()
        self._timestamps_ms.clear()
        self._sides.clear()
        self._tick_count = 0
        self._last_signal_tick = 0
        self._total_signals = 0
        self._buy_signals = 0
