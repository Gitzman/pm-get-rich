# L2 Order Book Fill Model — Assumptions

> Companion doc for `src/costs/book_fills.py`. Documents the assumptions,
> simplifications, and known limitations of the L2 fill simulator.

---

## Overview

The L2 fill simulator replaces the statistical fill model (`src/costs/fills.py`)
with a simulation that replays actual PMXT order book snapshots. Instead of
estimating fill probability from aggregate taker-volume and resting-depth
statistics, it tracks queue position through observed book state changes.

**Goal:** Produce realistic fill estimates for paper trading by using actual
market microstructure data rather than parameterized assumptions.

---

## Core Assumptions

### 1. Queue Position: Back of Queue (FIFO)

We assume our virtual order joins the **back of the queue** at its price level.
All existing resting depth at our price is ahead of us.

**Rationale:** Conservative. On Polymarket's CLOB, orders are filled in
price-time priority (FIFO). Since we're placing a hypothetical order, the
pessimistic assumption is that all existing orders were placed before ours.

**Alternative considered:** Mid-queue (0.5 × depth). Rejected because it
overstates fills — better to underestimate and be surprised.

### 2. Taker Flow Attribution (Interval Data)

Between consecutive book snapshots, we observe the **net change** in depth
at our price level. We attribute:

- **Net decrease → taker flow** consumed orders from the front of the queue.
  Our queue position advances by the decrease amount.
- **Net increase → new limit orders** joined behind us. No queue advancement.

**Known limitation:** This understates taker flow when new limit orders arrive
between snapshots, masking consumed volume. A 100→100 transition could mean
"0 consumed, 0 added" or "50 consumed, 50 added." We conservatively assume
the former.

**Impact:** Fill rates will be underestimated when there is significant limit
order placement between snapshots. This is intentional — phantom fills are
worse than missed fills for paper trading.

### 3. Binary Fill Model (No Partials)

An order either fills completely or not at all. Partial fills are not modeled.

**Rationale:** Same as the statistical model. Partial fills add complexity
without proportional benefit for strategy evaluation. If the queue drains
past our position, we count it as a full fill.

### 4. Marketable Orders = Immediate Fill

If an order crosses the spread at placement time (buy ≥ best ask, or
sell ≤ best bid), it fills immediately at our limit price.

**Rationale:** Our strategy uses maker (limit) orders, so this case is
uncommon. When it occurs, the order would execute as a taker.

### 5. Market Crossing = Guaranteed Fill

If any subsequent snapshot shows the market has crossed through our price
(best ask ≤ our buy price, or best bid ≥ our sell price), we count it as
a fill. The aggressive flow that moved the market would have to pass through
our price level, filling our resting order.

### 6. Fill Window

Orders expire if not filled within `fill_window_s` (default: 300 seconds).
This matches the statistical model's assumption.

**Rationale:** Limits the time we're exposed to the market. In practice, our
strategy places orders with GTC (Good Til Cancelled) but we evaluate fill
likelihood over a finite window.

### 7. Adverse Selection

Post-fill adverse selection cost uses the same model as `fills.py`:
`adverse_selection_ticks × tick_size × contracts`. The L2 simulator does
not (yet) compute realized post-fill drift from the book data.

**Future improvement:** Measure actual price drift over 30s/60s windows
after each simulated fill using subsequent book midpoints.

### 8. Exit Assumption

The exit leg (closing the position) is assumed to be a taker trade at
market price, paying taker fees. This matches the statistical model.

---

## Data Model

### PMXT Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| market_id | str | Polymarket condition_id |
| update_type | str | "book_snapshot" or "price_change" |
| data | str (JSON) | Book state payload |

### book_snapshot JSON

```json
{
  "best_bid": 0.55,
  "best_ask": 0.56,
  "bids": [[0.55, 100], [0.54, 50], ...],
  "asks": [[0.56, 80], [0.57, 60], ...],
  "timestamp": 1711360800000
}
```

- `bids`/`asks`: arrays of `[price, size]` pairs
- `timestamp`: epoch milliseconds (UTC)

### Snapshot Frequency

PMXT files are hourly. Within each file, snapshot frequency depends on
market activity — typically every few seconds for active markets.

---

## Known Limitations

1. **No order cancellation modeling:** We don't distinguish between depth
   decreases from taker fills vs. maker cancellations. Both advance our
   queue position, but cancellations don't represent real liquidity taking.

2. **No market impact:** Our virtual order does not affect the book.
   In reality, adding size at a price level increases depth and may
   discourage takers.

3. **Snapshot interpolation:** Between snapshots, we assume atomic
   transitions. Intra-snapshot microstructure is not modeled.

4. **No cross-price flow tracking:** We only track depth at our exact
   price level. Taker flow that sweeps multiple levels is only detected
   via the market-crossing check.

5. **Single-market isolation:** Each order is simulated independently.
   Correlated fills across markets are not modeled.

---

## Comparison to Statistical Model

| Aspect | Statistical (`fills.py`) | L2 (`book_fills.py`) |
|--------|--------------------------|----------------------|
| Fill probability | Estimated from aggregate stats | Simulated from actual book |
| Queue position | Parameterized fraction | Actual depth tracking |
| Taker volume | User-provided estimate | Observed from book changes |
| Data required | Scalar stats | PMXT L2 snapshots |
| Computation | O(1) per order | O(snapshots) per order |
| Output format | Compatible dict | Same + diagnostic fields |

---

## Test Coverage

Unit tests in `tests/test_book_fills.py` cover:

- Immediate fills (marketable orders)
- Queue position tracking across multiple snapshots
- Gradual queue drainage
- Depth increases (orders joining behind)
- Market crossing detection
- Edge cases: no data, empty book, zero depth, window expiry
- Drop-in interface compatibility with `fills.py`
- Multiple independent markets
- PMXT data parsing
