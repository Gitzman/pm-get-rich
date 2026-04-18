# L2 Fill Simulator Backtest Comparison (pm-gy0.8)

**Apples-to-apples**: same 5,532 TPP signals, same exit rules.
**Only change**: fill model swapped from statistical → L2 order book.

## Coverage (PMXT L2 data availability)

- **Signals with L2 book data:** 360 / 5,532 = **6.5%**
- **Signals without L2 data:** 5,172 (PMXT hourly parquets missing for those hours — not unfilled orders)
- **Fills on covered signals:** 360 / 360 = **100.0%** (true L2 fill rate when data is present)

All L2 metrics below are computed on the covered subset. Uncovered signals cannot be evaluated and are excluded from P&L, CI, hit rate, etc. (they are not "unfilled" — the data is simply absent).

## Headline: Per-trade P&L 95% CI excludes zero? **YES (NEGATIVE)**
  - CI: [-28.7112, -19.6300]
  - Mean P&L per filled trade: -24.1368
  - Based on 360 filled trades drawn from 360 covered signals (6.5% of 5,532).

## Side-by-Side: Old Fill Model vs L2 Simulator

| Metric | Conservative | Moderate | Optimistic | **L2 Book** |
|--------|-------------|----------|------------|-------------|
| Signals | 5532 | 5532 | 5532 | **5532** |
| Fills | 922.0 | 5532.0 | 5532.0 | **360.0** |
| Fill Rate (of all) | 0.167 | 1.000 | 1.000 | **0.065** |
| Mean P&L/Fill | 0.8570 | 1.3570 | 1.6070 | **-24.1368** |
| Total P&L | 790.14 | 7506.81 | 8889.81 | **-8689.24** |
| Win Rate | 0.288 | 0.321 | 0.330 | **0.158** |
| Hit Rate | 0.541 | 0.541 | 0.541 | **0.297** |
| Signals w/ L2 data | - | - | - | **360** |
| Fill Rate (of covered) | - | - | - | **1.000** |
| P&L 95% CI | - | - | - | **[-28.7112, -19.6300]** |

## Adverse Selection Drift (L2 Fills Only)

| Window | Mean Adverse Drift |
|--------|-------------------|
| 30s | 0.223133 |
| 60s | 0.198267 |
| 120s | 0.273300 |

## Fill Rate by Market

_20 events had L2 coverage; 204 events had NO PMXT data (listed separately)._

### Top 10 Covered Events by Fill Rate
| Event | City | Signals | Covered | Fills | Fill Rate (of covered) |
|-------|------|---------|---------|-------|------------------------|
| 322401 | Tokyo | 24 | 24 | 24 | 1.000 |
| 322407 | Warsaw | 24 | 24 | 24 | 1.000 |
| 322394 | Atlanta | 24 | 24 | 24 | 1.000 |
| 322424 | Los Angeles | 24 | 24 | 24 | 1.000 |
| 322420 | Shenzhen | 24 | 24 | 24 | 1.000 |
| 322406 | Madrid | 24 | 24 | 24 | 1.000 |
| 322405 | Milan | 24 | 24 | 24 | 1.000 |
| 322397 | Wellington | 24 | 24 | 24 | 1.000 |
| 322388 | Buenos Aires | 24 | 24 | 24 | 1.000 |
| 322403 | Shanghai | 24 | 24 | 24 | 1.000 |

### Bottom 10 Covered Events by Fill Rate
| Event | City | Signals | Covered | Fills | Fill Rate (of covered) |
|-------|------|---------|---------|-------|------------------------|
| 322396 | Chicago | 24 | 12 | 12 | 1.000 |
| 322331 | Ankara | 24 | 12 | 12 | 1.000 |
| 322410 | Beijing | 24 | 12 | 12 | 1.000 |
| 322402 | Hong Kong | 24 | 12 | 12 | 1.000 |
| 322422 | Denver | 24 | 12 | 12 | 1.000 |
| 322411 | Wuhan | 24 | 12 | 12 | 1.000 |
| 322408 | Taipei | 24 | 12 | 12 | 1.000 |
| 322404 | Singapore | 24 | 12 | 12 | 1.000 |
| 322398 | Lucknow | 24 | 12 | 12 | 1.000 |
| 322395 | Miami | 24 | 12 | 12 | 1.000 |

### Events with NO PMXT Coverage (204)
_These had zero L2 snapshots — data absence, not 0% fill._

| Event | City | Signals |
|-------|------|---------|
| 306333 | Ankara | 24 |
| 295990 | Ankara | 24 |
| 289202 | Ankara | 24 |
| 299422 | Ankara | 24 |
| 292571 | Ankara | 24 |
| 302804 | Ankara | 24 |
| 306330 | Atlanta | 28 |
| 299419 | Atlanta | 24 |
| 292568 | Atlanta | 24 |
| 289199 | Atlanta | 24 |
| 302801 | Atlanta | 24 |
| 295987 | Atlanta | 24 |
| 306351 | Austin | 24 |
| 296015 | Austin | 24 |
| 302822 | Austin | 24 |
| 299440 | Austin | 24 |
| 296004 | Beijing | 24 |
| 292585 | Beijing | 24 |
| 299436 | Beijing | 24 |
| 306347 | Beijing | 24 |
| 302818 | Beijing | 24 |
| 306321 | Buenos Aires | 24 |
| 295978 | Buenos Aires | 24 |
| 302792 | Buenos Aires | 24 |
| 289190 | Buenos Aires | 24 |
| _(+179 more)_ | | |
