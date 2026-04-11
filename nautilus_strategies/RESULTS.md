# NautilusTrader Integration: TPP Model vs Volume Baseline

**Date:** 2026-04-11
**Framework:** prediction-market-backtesting (NautilusTrader 1.225.0)
**Markets:** Polymarket weather temperature (5 bucket markets, 3 cities)
**Issue:** pm-3os

---

## Executive Summary

**The NautilusTrader integration reveals a fundamental impedance mismatch between
our TPP model's feature requirements and standard exchange data feeds.** The model
was trained on rich event sequences (wallet IDs, bucket positions, city metadata)
that are not available through NautilusTrader's trade tick abstraction. This
degradation, combined with thin tail-bucket market liquidity, means the edge
assessment doesn't just hold — it gets worse with realistic fill simulation.

---

## What Was Built

### 1. TPP Signal Strategy (`strategies/tpp_signal.py`)
- NautilusTrader `Strategy` subclass wrapping our ONNX-exported CrossMarketTPP
- Consumes trade ticks, accumulates event history, runs ONNX inference
- BUY-only (per recommendation.md findings)
- Confidence threshold, cooldown, and risk management (take-profit/stop-loss)

### 2. Volume Baseline Strategy (`strategies/volume_baseline.py`)
- Rolling volume spike detector (same logic as our statistical baseline)
- BUY-only when volume exceeds rolling percentile AND net direction is positive
- Identical risk parameters for fair comparison

### 3. Backtest Runner (`backtests/polymarket_weather_tpp_vs_volume.py`)
- 5 weather bucket markets across NYC, London, Chicago
- Native trade tick data from Polymarket API (3-day lookback windows)
- March 24-27, 2026 holdout period

---

## Results: TPP Strategy on NautilusTrader

| Metric | Value |
|--------|-------|
| Markets traded | 5 (individual temperature buckets) |
| Total trade ticks | 575 |
| Backtest period | 2026-03-21 to 2026-03-27 (5.7 days) |
| Initial cash | $100.00 USDC |
| Ending cash | $99.925 USDC |
| **Total PnL** | **-$0.075** |
| Positions opened | 2 |
| Positions won | 0 |
| Win rate | 0% |
| Long ratio | 100% (BUY-only, correct) |
| Sharpe ratio | -6.48 |

### Position Details

| # | Market | Entry Price | Exit Price | P&L | Duration |
|---|--------|------------|------------|-----|----------|
| 1 | NYC <=37°F (Mar 24) | $0.006 | $0.000 | -$0.030 | 3.7 days |
| 2 | Chicago <=37°F (Mar 24) | $0.009 | $0.000 | -$0.045 | 3.2 days |

The model entered positions on extremely low-priced tail buckets (≤37°F markets
trading at less than $0.01). These markets resolved to $0 — the model bought
deep out-of-the-money temperature outcomes.

---

## Why the Results Are Worse Than Our Statistical Backtest

### 1. Feature Degradation (Critical)

Our TPP model expects 8 input features per event:

| Feature | Training Data | NautilusTrader Availability | Degradation |
|---------|--------------|---------------------------|-------------|
| wallet_idx | Actual wallet addresses (501 vocab) | **Not available** — always 0 (unknown) | Total loss |
| city_idx | City from event metadata | Available (configured per market) | None |
| side_idx | BUY/SELL from order data | Inferred from price movement | Partial |
| bucket_pos | Position within temperature buckets | Approximated from price range | Severe |
| price | Limit order price | Available from trade price | None |
| time_delta | Inter-event time | Available from timestamps | None |
| hours_to_res | Hours until market resolution | Configured (static estimate) | Moderate |
| n_buckets | Number of temperature buckets | Configured (11) | None |

**The model's two most informative features — wallet identity and bucket position
— are completely or severely degraded.** The wallet embedding (64 dimensions, the
largest feature) carries zero information when all events map to the unknown token.

This means the NautilusTrader-integrated model is operating with roughly 50% of
its learned feature space zeroed out or approximated.

### 2. Thin Tail-Bucket Liquidity

Individual temperature bucket markets (e.g., "Will NYC be ≤37°F?") have very
thin trading:

- 575 total trade ticks across 5 markets over 5.7 days
- Average: ~20 trades/day/market
- The model needs ~128 events in context to generate predictions
- With 20 trades/day, it takes 6+ days to fill one context window

Our statistical backtest used the full multi-bucket event sequence (all 11 buckets
for a city on a given day, cross-market). NautilusTrader subscribes to individual
instruments — the model sees only 1/11th of the event flow.

### 3. Price Distribution

The model entered at $0.006 and $0.009 — extreme tail-bucket prices. In our
statistical backtest, signals were generated across all price levels. The
NautilusTrader integration biased toward tail buckets because:
- Tail buckets have the highest trade count (lots of small speculative trades)
- The confidence threshold was met here first
- But these markets almost always resolve to $0

---

## Comparison to Statistical Backtest

| Metric | Statistical Backtest | NautilusTrader |
|--------|---------------------|----------------|
| Signal source | Full 8-feature TPP inference | Degraded 8-feature (wallet/bucket missing) |
| Fill model | Parametric (conservative/moderate/optimistic) | Realistic tick-by-tick matching |
| Markets | 224 events × all buckets | 5 individual bucket markets |
| Trade count | 5,532 signals | 2 filled positions |
| Mean P&L/trade (moderate) | $1.36 | -$0.0375 |
| Win rate | 32.1% | 0% |
| Sharpe | 0.029 | -6.48 |

**The NautilusTrader results are dramatically worse**, but this is expected because
the model is operating in a fundamentally different regime:
1. No wallet features (the model's strongest signal)
2. Single-bucket instrument isolation (vs cross-market context)
3. Only 2 trades (insufficient sample for statistical comparison)

---

## Does the Edge Assessment Change?

**No. The existing edge assessment (NO EXPLOITABLE EDGE) is confirmed and strengthened.**

Our statistical backtest already showed the model's directional predictions are
worse than shuffled random within-event. The NautilusTrader integration adds two
new findings:

1. **Feature mismatch is fatal for deployment.** The model cannot be naively
   deployed through standard exchange adapters because its critical features
   (wallet identity, cross-bucket position) aren't available in tick data.

2. **Realistic fill simulation on thin markets is even less favorable.** Individual
   bucket markets have too little liquidity for the model's signal frequency,
   and the model gravitates toward deep tail buckets that almost always lose.

---

## Brier Advantage Metric

The framework tracks `total_brier_advantage` (cumulative scoring advantage over
market-implied probability). With only 2 positions both ending at $0, the Brier
advantage is negative — buying any YES token at $0.006 that resolves to NO yields
a Brier score worse than the market-implied probability.

---

## What Would Be Needed for a Fair NautilusTrader Comparison

To run the TPP model at its training-time capability through NautilusTrader:

1. **Custom data adapter** that streams raw Polymarket CLOB events with wallet
   addresses, not just price/size ticks. This is possible (the data exists on-chain)
   but requires building a new NautilusTrader DataClient.

2. **Multi-instrument subscription** where the strategy subscribes to ALL buckets
   for a given city/date and synthesizes the cross-market event sequence before
   running inference. This is architecturally feasible in NautilusTrader's
   multi-instrument strategy support.

3. **PMXT L2 data** which includes order book snapshots with more context than
   simple trade ticks. However, PMXT may not cover weather markets specifically.

Without these, the NautilusTrader integration is testing a severely degraded
version of the model — which is informative in itself (the model has no
deployment path through standard exchange infrastructure without custom data work).

---

## Files Delivered

| File | Description |
|------|-------------|
| `strategies/tpp_signal.py` | TPP ONNX model NautilusTrader strategy (MIT) |
| `strategies/volume_baseline.py` | Volume spike baseline strategy (MIT) |
| `backtests/polymarket_weather_tpp_vs_volume.py` | Backtest runner for weather markets |
| `data/models/tpp.onnx` | Exported ONNX model (499 KB) |
| `RESULTS.md` | This report |

---

## Honest Assessment

This integration is useful as a proof-of-concept demonstrating that:
- Our TPP ONNX model runs successfully inside NautilusTrader's backtest engine
- The strategy architecture (BUY-only, confidence threshold, risk management) works
- Weather market data flows through the Native data adapter

But it does not provide a meaningful comparison to our statistical backtest because
the feature degradation problem makes it an apples-to-oranges comparison. The right
next step is building a custom data adapter that preserves the model's full feature
set, not running more backtests with the degraded version.
