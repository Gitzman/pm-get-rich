# Convoy 5: Honest Assessment — Does the Edge Survive Realistic Fills?

**Date:** 2026-04-15
**Bead:** pm-gy0.6
**Verdict:** The TPP edge does not survive realistic fills. Volume is untested under L2. Recommend proceeding on volume only, contingent on running it through L2.

---

## 1. Under realistic fills, does the TPP strategy have edge?

**No.**

| Metric | Statistical Model (moderate) | L2 Order Book |
|--------|------------------------------|---------------|
| Mean P&L per filled trade | $1.36 | **-$23.41** |
| 95% CI | [-0.15, 4.11] | **[-27.96, -18.86]** |
| Hit rate (direction correct) | 54.1% | **29.7%** |
| Win rate (P&L > 0) | 32.1% | **15.8%** |
| Total P&L | $7,507 | **-$8,689** |

The CIs do not overlap. Not close. The statistical fill model produced an illusory edge; the L2 order book model reveals systematic losses.

Both buy and sell signals lose money under L2 fills:
- **BUY** (264 fills): mean P&L = -$25.05, direction correct 26.3% of non-flat
- **SELL** (96 fills): mean P&L = -$18.89, direction correct 41.2% of non-flat

The previous recommendation (recommendation.md) said "Paper trade TPP, BUY-only." That recommendation is now dead. BUY fills are the *worst* performers under realistic fills, not the best — the BUY/SELL asymmetry inverted when switching from statistical to L2 fills.

## 2. Was the volume baseline run through L2?

**No.** The volume baseline (485 signals, 36 configurations) was evaluated using the statistical fill model only (conservative/moderate/optimistic regimes). It has not been tested against L2 order book data.

Under the statistical model, volume beats random in 10 of 12 moderate-regime configurations. Best config: 60s window, 60s dt, moderate — mean expected P&L $4.01/trade, Sharpe 0.082, hit rate 55.4%. This is a stronger signal than TPP ever showed under the same model.

But we now know the statistical model is unreliable. Volume's edge under statistical fills could evaporate the same way TPP's did. **We cannot assess volume's viability without running it through L2.**

## 3. CI comparison — what happened?

| Model | 95% CI per trade | Notes |
|-------|-------------------|-------|
| Statistical (moderate, 60s) | [-0.15, +4.11] | Crossed zero marginally |
| L2 Order Book | [-27.96, -18.86] | Excludes zero on negative side |

The gap is ~$20-30 per trade. These are not the same distribution. The fill model was not a minor assumption — it was the load-bearing wall. When it collapsed, everything above it fell.

The statistical model assumed fill probability was independent of market direction. It is not.

## 4. What killed the edge?

**Adverse selection on fills.**

The statistical fill model assigned fill probability uniformly across signals regardless of market conditions. Under this assumption, the model's 54.1% directional accuracy (which itself barely beat random) produced a thin but positive expected P&L.

The L2 order book reveals the actual mechanism: orders fill precisely when the market is moving against the prediction. Filled trades have 29.7% direction accuracy versus 54.1% overall. The market isn't moving randomly around our fills — it's moving *against* them, systematically.

Adverse drift confirms this:

| Window after fill | Mean adverse drift |
|-------------------|--------------------|
| 30s | 0.223 |
| 60s | 0.198 |
| 120s | 0.273 |

Positive adverse drift at all windows = price consistently moves against our position after fill.

**Why this happens:** Limit orders on the wrong side of informed flow get picked off. When a genuine directional move starts, our resting order on the wrong side fills immediately (taker flow sweeps it). Our orders on the right side sit in the queue (no one is crossing the spread to trade against the move). The fill itself is a signal that we're wrong — and the L2 data captured this. The statistical model couldn't.

---

## Recommendation

Of the three options from the Adam brief:

| Option | Assessment |
|--------|-----------|
| Proceed to paper trading (both strategies have edge) | **Ruled out.** TPP is dead. Volume is unproven under L2. |
| Proceed on volume only (TPP killed, volume survives) | **Conditional.** Volume must pass L2 first. |
| Stop and pivot (both killed, book-event model becomes next path) | **Premature.** Volume hasn't been tested. |

**Recommended path: Run volume baseline through L2 fill simulator before any further decisions.**

Volume has better fundamentals than TPP under statistical fills (beats random in 10/12 configs vs TPP losing to its own shuffled baseline). But the same adverse selection mechanism could apply. The L2 test is now the gate — no strategy advances to paper trading without passing it.

If volume also fails L2: stop. The timing-signal approach is dead and the book-event model is the next path.

If volume passes L2: proceed to paper trading with volume-only signals. The TPP model's contribution is zero.

**The numbers say what they say. The TPP edge was an artifact of the fill model.**
