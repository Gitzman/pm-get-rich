# Does the Edge Exist?

**Verdict: NO.** The directional signal has no exploitable edge after costs.

---

## The Numbers

| Metric (moderate fill regime) | Model | Shuffled Baseline | Random Baseline |
|-------------------------------|-------|-------------------|-----------------|
| Mean expected P&L per trade   | $1.36 | **$2.20**         | -$1.28          |
| Total expected P&L (5,532 signals) | $7,507 | **$12,176** | -$7,080         |
| Sharpe ratio                  | 0.029 | **0.047**         | -0.027          |
| Hit rate (directional)        | 54.1% | 53.2%             | 49.7%           |
| Win rate (P&L > 0)            | 32.1% | 38.6%             | 29.3%           |

**The shuffled baseline beats the model in 10 of 12 parameter configurations.**
This holds across all three fill regimes (conservative, moderate, optimistic).

## Where It Died

**The model's directional predictions are worse than random within-event.**

The shuffled baseline keeps the model's timing (when to trade) but randomizes which
signal gets which prediction within each event. It consistently outperforms the model.
This means the model's specific directional assignments actively destroy value.

**Root causes:**

1. **Extreme long bias.** The model predicts long 86.5% of the time. Actual outcomes
   are 37% up / 33% flat / 31% down. The model barely distinguishes direction —
   max confidence is only |pred - 0.5| = 0.19, mean = 0.10.

2. **Prediction confidence is noise-level.** The predicted bucket positions cluster
   tightly around 0.57 (mean). The model learned a slight positive bias, not a
   meaningful directional signal.

3. **Gross P&L is meager.** Even before costs, mean gross P&L is $3.14/trade
   (100 contracts). After exit fees (~$0.50) and adverse selection ($0.50-$1.00/trade),
   most of the edge evaporates. Win rate after costs: 32%.

4. **Model is a good forecaster, not a good trading signal.** The cross-market
   transformer achieves LL = 10.5 (strong) on held-out events. It models trade
   sequences well. But predicting the next wallet/price/timing in a sequence
   doesn't translate into directional price prediction. These are different tasks.

## What the Model CAN Do

The model has genuine forecasting ability:
- Held-out log-likelihood: 10.50 (strong, generalizes to unseen cities)
- Beats classical Hawkes baseline where both were tested
- The "when to trade" component has value (beats fully-random baseline)

The TPP is good at modeling market microstructure. It just can't predict
which way the price will move.

## Capacity (Moot, But Noted)

At 100 contracts: fill probability is ~1.0 (moderate regime).
At 1,000+ contracts: fill probability drops, but this is irrelevant given no edge.
Max theoretical capacity: ~$10K total P&L at 100 contracts — tiny.

## Best Parameters (If You Had to Pick)

dt=30s, threshold 5%, moderate fill: model gets $1.57/trade, Sharpe 0.033.
This is the only region where the model beats shuffled in 2/12 configs.
The edge is ~$1.50/trade, CI crosses near zero, and it's 2 of 12 configs.
Not enough to trade on.

## Next Steps

1. **Don't deploy this signal.** The directional component has negative alpha.
2. **Investigate the timing signal.** The model picks meaningful entry times
   (beats random). Could a separate directional model be layered on top?
3. **Consider different target variable.** The TPP predicts sequence dynamics,
   not price direction. A model trained directly on price_change might fare better.
4. **More data.** 5,532 signals across 224 events is thin. But adding data
   won't fix a model architecture mismatch.
