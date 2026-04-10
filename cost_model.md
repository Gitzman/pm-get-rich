# Cost Model: Polymarket Maker Strategy

> **Convoy 1 deliverable (pm-kxg.1).** Must be reviewed before Convoy 2 begins.

## Fee Structure

### Taker Fee

```
fee = θ × C × p × (1 − p)
```

| Variable | Meaning |
|----------|---------|
| θ (theta) | Fee coefficient, varies by category |
| C | Number of contracts (shares) |
| p | Limit price ∈ [0.01, 0.99] |

The fee is symmetric around p = 0.50 and approaches zero at price extremes.
Maximum fee per contract at p = 0.50: `θ × 0.25`.

### Theta by Market Category

| Category | θ | Maker Rebate |
|----------|------|-------------|
| Crypto | 0.072 | 20% |
| Sports | 0.030 | 25% |
| Finance | 0.040 | 25% |
| Politics | 0.040 | 25% |
| Economics | 0.050 | 25% |
| Culture | 0.050 | 25% |
| **Weather** | **0.050** | **25%** |
| Other | 0.050 | 25% |
| Mentions | 0.040 | 25% |
| Tech | 0.040 | 25% |
| Geopolitics | 0.000 | — |

**Our target: weather markets, θ = 0.05.**

### Maker Fees

**Makers pay zero fees.** This is the primary reason we use a maker strategy.

### Maker Rebate

Makers receive a daily USDC rebate from a pool equal to 25% of taker fees
collected in their market. Distribution is pro-rata by fee-equivalent volume:

```
rebate_i = (fee_equiv_i / Σ fee_equiv_j) × rebate_pool
```

where `fee_equiv = θ × C × p × (1 − p)` computed as if the maker fill were
a taker trade.

**For cost modeling, we OMIT the rebate.** It's pooled, unpredictable, and
omitting it makes our cost estimates conservative.

### Gas

Effectively $0. Polymarket uses Polygon meta-transactions — users don't pay gas.

### Price Grid

- Tick size: $0.01
- Price range: $0.01 to $0.99
- Fee precision: rounded to 5 decimal places

### Fee Examples (Weather, θ = 0.05)

| Price | Fee per contract |
|-------|-----------------|
| 0.10 | 0.0045 |
| 0.25 | 0.009375 |
| 0.50 | 0.0125 |
| 0.75 | 0.009375 |
| 0.90 | 0.0045 |

### Round-Trip Cost (Maker Entry, Taker Exit)

For our strategy — enter via limit order (maker, $0 fee), exit at resolution
or via market order (taker fee):

```
round_trip_cost = 0 (entry) + θ × C × p_exit × (1 − p_exit)
```

At resolution, p_exit → 1.0 or 0.0, so the exit taker fee → $0. This means
**if we hold to resolution, our fee cost is effectively zero.**

If we exit early via market order at price p_exit, the fee applies.

### API Verification

Always fetch the current fee rate dynamically:
```
GET https://clob.polymarket.com/fee-rate?token_id={token_id}
```
Returns `fee_rate_bps` to include in signed order payloads. Do not hardcode.

Sources:
- docs.polymarket.com/trading/fees (verified 2026-04-10)
- polymarketexchange.com/fees-hours.html (verified 2026-04-10)

---

## Fill Model

### Overview

We place limit (maker) orders and wait for taker flow to fill us. The key
question: **given a limit order at price P, does it fill within a useful
time window?**

### Fill Probability Model

```
P(fill) = min(1, taker_volume / (queue_ahead + our_size))
```

Where:
- `taker_volume`: historical taker volume at price P within the fill window
- `queue_ahead`: resting depth at P × queue_position_fraction
- `our_size`: our order size in contracts

### Assumptions (Conservative)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Fill window | 300s (5 min) | Typical holding period for short-horizon signals |
| Queue position | 1.0 (last) | We assume all existing depth fills before us |
| Min taker volume | 10 contracts | Below this, fills are too sparse to model |
| Fill granularity | Binary (full or nothing) | No partial fills — conservative |

**Queue position = 1.0 (last in queue)** is the critical conservative assumption.
In practice, a maker who posts early may get mid-queue priority, but we can't
observe queue position from historical data, so we assume worst case.

### Adverse Selection

When a limit order fills, it's often because an informed taker moved the price
through our level. This creates **adverse selection**: the market price after
our fill tends to move against our position.

We measure adverse selection as the post-fill price drift:

```
adverse_selection = |price(t + Δ) − price(t_fill)|
```

Over windows Δ = 30s and 60s.

**Default assumption:** 0.5 ticks of adverse drift per fill.
At $0.01/tick, this is $0.005 per contract.

This is a placeholder until we can measure actual post-fill drift from
historical order book data (Convoy 3+).

### Expected All-In Cost per Fill

For a maker limit order that fills:

```
total_cost = entry_fee + exit_fee + adverse_selection
           = 0 + θ×C×p×(1-p) + 0.5 × tick × C
```

At p = 0.50 (worst case for fees), C = 100 contracts:
- Entry fee: $0.00
- Exit fee (at resolution): $0.00 (p → 1.0 or 0.0)
- Exit fee (early, at p = 0.50): $1.25
- Adverse selection: $0.50
- **Total (hold to resolution): $0.50**
- **Total (early exit at 0.50): $1.75**

### Cost as Fraction of Edge

For a $100 position:

| Scenario | Cost | Edge needed to profit |
|----------|------|-----------------------|
| Hold to resolution | $0.50 | 0.5% |
| Early exit at p=0.50 | $1.75 | 1.75% |
| Early exit at p=0.25 | $1.44 | 1.44% |

**Conclusion:** If our signal has >2% edge, trading costs are manageable.
The maker strategy is significantly cheaper than taker execution.

---

## Implementation

Code: `src/costs/`

```python
from src.costs import taker_fee, round_trip_cost, expected_fill_cost

# Fee for 100 contracts at p=0.50, weather market
fee = taker_fee(price=0.50, contracts=100, theta=0.05)
# => 1.25

# Round trip: maker entry at 0.50, taker exit at resolution (p→1.0)
rt = round_trip_cost(entry_price=0.50, exit_price=0.99, contracts=100)
# => 0.0495 (tiny fee near price extremes)

# Full cost model with fill assumptions
cost = expected_fill_cost(
    price=0.50,
    contracts=100,
    taker_volume_at_price=500,
    resting_depth_at_price=200,
)
# => {'fill_prob': 0.3333, 'entry_fee': 0, 'exit_fee': 1.25,
#     'adverse_selection': 0.5, 'total_cost': 1.75, 'expected_cost': 0.58333}
```

---

## Open Questions for Convoy 2+

1. **Empirical adverse selection**: Measure actual post-fill drift from
   historical trade data. The 0.5-tick default may be too high or too low.
2. **Queue position**: Can we infer queue position from order book snapshots?
   Would improve fill probability estimates.
3. **Category-specific theta**: Should we fetch theta per-market via the API
   rather than hardcoding? Some markets may have custom rates.
4. **Maker rebate modeling**: Currently omitted. If rebates are material
   (>10% of costs), we should estimate them.
5. **Partial fills**: Current model is binary. Real fills may be partial —
   worth modeling if order sizes are large relative to taker flow.
