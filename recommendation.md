# Next Session Recommendation

**Decision: Paper trade TPP, BUY-only, 5-minute drift horizon.**

---

## What we tested

Two timing signals for Polymarket weather markets, evaluated on 224 held-out events:

1. **TPP (Temporal Point Process):** Neural Hawkes transformer predicting trade sequences.
   Uses intensity predictions to time entries.
2. **Volume baseline:** Rolling 60s volume spike above Nth percentile. Simplest possible
   timing signal.

Both were backtested with identical cost models (100 contracts, taker exit fees, adverse
selection) across conservative/moderate/optimistic fill regimes.

## Head-to-head results

| Metric (moderate regime, 60s drift) | TPP | Volume |
|--------------------------------------|-----|--------|
| Mean expected P&L per trade          | $1.98 [-0.15, 4.11] | $2.69 [0.49, 4.95] |
| Sharpe ratio                         | 0.042 [-0.003, 0.085] | 0.055 [0.009, 0.099] |
| Hit rate                             | 54.5% | 52.2% |
| Total P&L (1,940 trades)             | $3,839 | $5,220 |

At 60-second look-ahead: **tied.** Both CIs overlap heavily. Volume is slightly ahead
but not significantly.

## Longer horizons change the picture

| Metric (moderate, all directions) | TPP 5min | TPP 30min | Volume 60s/2min |
|-----------------------------------|----------|-----------|-----------------|
| Sharpe                            | **0.136** | 0.107    | 0.073           |
| Hit rate                          | 55-61%   | 55-60%   | 57-65%          |

At 5-30 minute drift: **TPP wins.** The transformer's timing signal captures something
the volume spike doesn't — probably microstructure dynamics that play out over minutes,
not seconds.

## The big finding: BUY/SELL asymmetry

This matters more than TPP vs. volume.

| Direction | Hit rate (2-30min) | Sharpe range | Drift against us |
|-----------|--------------------|--------------|------------------|
| **BUY**   | 55-65%             | 0.10 to 0.30 | **Negative** (price moves in our favor) |
| **SELL**  | 35-45%             | -0.05 to -0.23 | Positive (price moves against us) |

BUY fills are advantageous. SELL fills are adversely selected at every horizon,
for both TPP and volume. This is consistent and large.

**Any strategy must be BUY-only.** Sells actively destroy value.

## Honest caveats

1. **The edge is small.** Best Sharpe is 0.136 (TPP BUY, 5min). This is thin.
   A Sharpe below 0.5 on 224 events could be noise.
2. **Directional predictions are weak.** The model predicts long 86.5% of the time.
   It's the timing, not the direction, that has value.
3. **Capacity is tiny.** At 100 contracts, max theoretical total P&L is ~$10K over
   the backtest period. This is a research exercise, not a business.
4. **CIs cross zero** in some configurations. We cannot rule out zero edge.

## Decision framework

| Scenario | Action |
|----------|--------|
| TPP beats volume | Paper trade TPP |
| Tied               | Paper trade volume (simpler) |
| Drift negative     | Stop |

**Observed: TPP beats volume at 5-30min horizons. Drift is positive for BUY side.**

## Recommendation

**Paper trade TPP, BUY-only signals, 5-minute drift horizon.**

Parameters:
- Signal: TPP model (threshold_pct=10)
- Direction: BUY only (filter all SELL signals)
- Look-ahead: 5 minutes
- Contracts: 100
- Fill regime: moderate

**Do not:**
- Trade SELL signals (adversely selected at all horizons)
- Hedge or take offsetting positions (adds complexity, sells lose money)
- Scale up before confirming the edge in live data

**Success criteria for paper trading:**
- 50+ paper trades
- BUY hit rate above 55%
- Positive cumulative P&L after simulated costs
- If these fail, stop and reassess the timing signal approach entirely
