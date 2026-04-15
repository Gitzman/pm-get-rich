# L2 Fill Simulator Backtest Comparison (pm-gy0.4)

**Apples-to-apples**: same 5,532 TPP signals, same exit rules.
**Only change**: fill model swapped from statistical -> L2 order book.

## COVERAGE LIMITATION

PMXT order book data only covers March 30 events at signal time. Of 5,532 signals:
- **360 signals (6.5%)** had L2 book data -> ALL 360 filled (100% fill rate with data)
- **5,172 signals (93.5%)** had NO L2 book data (PMXT coverage gap, not unfilled)
- **20 unique events** (all from March 30, event_ids 322xxx) had coverage
- The 0% fill rate for earlier events is data absence, NOT market illiquidity

**Headline numbers below reflect only the 360 data-covered signals for L2 fills.**

## Headline: Per-trade P&L 95% CI excludes zero? **YES (NEGATIVE)**
  - CI: [-28.7112, -19.6300]
  - Mean P&L per filled trade: -24.1368
  - **The CI excludes zero on the NEGATIVE side. Filled trades are losing money.**
  - Direction accuracy for filled trades: 29.7% (66/222 non-flat)
  - Mean gross P&L per fill: -23.4108 (before fees)

## Side-by-Side: Old Fill Model vs L2 Simulator

| Metric | Conservative | Moderate | Optimistic | **L2 Book** |
|--------|-------------|----------|------------|-------------|
| Signals | 5532 | 5532 | 5532 | **5532** |
| Signals w/ data | 5532 | 5532 | 5532 | **360** |
| Fills | 922.0 | 5532.0 | 5532.0 | **360** |
| Fill Rate (all) | 0.167 | 1.000 | 1.000 | **0.065** |
| Fill Rate (w/ data) | - | - | - | **1.000** |
| Mean P&L/Fill | 0.8570 | 1.3570 | 1.6070 | **-24.1368** |
| Total P&L | 790.14 | 7506.81 | 8889.81 | **-8689.24** |
| Win Rate | 0.288 | 0.321 | 0.330 | **0.158** |
| Hit Rate | 0.541 | 0.541 | 0.541 | **0.297** |
| P&L 95% CI | - | - | - | **[-28.7112, -19.6300]** |

## Key Observations

1. **All signals with L2 data filled (100%)**. The orders are marketable or find
   enough taker flow within the fill window. Fill rate is not the problem.

2. **Direction accuracy collapses on filled trades (29.7% vs 54.1% overall)**.
   The signals that fill in the real book tend to be wrong about direction.
   This is classic adverse selection: we get filled precisely when the market
   is moving against our prediction.

3. **Both buy and sell signals lose money**: Buy mean P&L = -25.05, Sell = -18.89.
   The loss is not side-specific.

4. **The statistical fill model was masking losses**. By assuming uniform fill
   probability independent of market conditions, the old model spread P&L across
   signals that wouldn't actually fill (winners and losers alike). The L2 model
   reveals which signals ACTUALLY execute -- and they're the losers.

## Adverse Selection Drift (L2 Fills Only)

| Window | Mean Adverse Drift |
|--------|-------------------|
| 30s | 0.223133 |
| 60s | 0.198267 |
| 120s | 0.273300 |

Adverse drift is consistently positive (0.20-0.27), confirming that filled
orders face systematic adverse price movement.

## Fill Rate by Market (Top 10 / Bottom 10)

### Highest Fill Rate (All March 30 Events)
| Event | City | Signals | Fills | Fill Rate |
|-------|------|---------|-------|-----------|
| 322388 | Buenos Aires | 24 | 24 | 1.000 |
| 322424 | Los Angeles | 24 | 24 | 1.000 |
| 322394 | Atlanta | 24 | 24 | 1.000 |
| 322420 | Shenzhen | 24 | 24 | 1.000 |
| 322403 | Shanghai | 24 | 24 | 1.000 |
| 322405 | Milan | 24 | 24 | 1.000 |
| 322397 | Wellington | 24 | 24 | 1.000 |
| 322407 | Warsaw | 24 | 24 | 1.000 |
| 322406 | Madrid | 24 | 24 | 1.000 |
| 322401 | Tokyo | 24 | 24 | 1.000 |

### Lowest Fill Rate (No L2 Data - PMXT Coverage Gap)
| Event | City | Signals | Fills | Fill Rate |
|-------|------|---------|-------|-----------|
| 306309 | Paris | 24 | 0 | 0.000 |
| 302790 | Sao Paulo | 24 | 0 | 0.000 |
| 299430 | Singapore | 24 | 0 | 0.000 |
| 299410 | Buenos Aires | 24 | 0 | 0.000 |
| 295988 | Miami | 24 | 0 | 0.000 |
| 292585 | Beijing | 24 | 0 | 0.000 |
| 292559 | Buenos Aires | 24 | 0 | 0.000 |
| 306347 | Beijing | 24 | 0 | 0.000 |
| 306325 | Toronto | 24 | 0 | 0.000 |
| 302794 | Seoul | 32 | 0 | 0.000 |

## Methodology Notes

- Test window: March 25-31, 2026
- PMXT coverage: March 30 events only (events 322xxx)
- Signal source: data/signals/signals.parquet (5,532 signals, unchanged from previous runs)
- L2 fill simulator: src/costs/book_fills.py (queue position tracking, snapshot-by-snapshot)
- Old fill model: src/costs/fills.py (statistical, fill_probability * net_pnl)
- Exit: time-stop (same as previous backtests)
- Contracts: 100 per signal
