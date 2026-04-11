# Architecture Note: Polymarket Rust Paper Trading System

> **Convoy 1 deliverable (pm-38v.1).** Must be reviewed before Convoy 2 begins.
> Covers: L1/L2 auth, WSS channels, rate limits, tick rules, maker rebate,
> settlement, rs-clob-client assessment, NautilusTrader evaluation.

---

## 1. Polymarket CLOB API Overview

Polymarket runs a Central Limit Order Book (CLOB) on Polygon. All trading goes
through the CLOB REST API (`clob.polymarket.com`) and WebSocket feed
(`ws-subscriptions-clob.polymarket.com`). Positions are conditional tokens
(ERC-1155) on the CTF (Conditional Token Framework) exchange contracts.

**Base URLs:**
- REST: `https://clob.polymarket.com`
- WSS: `wss://ws-subscriptions-clob.polymarket.com`
- Gamma (market discovery): `https://gamma-api.polymarket.com`
- Data (analytics): `https://data-api.polymarket.com`

---

## 2. Authentication: L1/L2

### L1: Initial Credential Derivation

L1 uses EIP-712 typed data signing with an Ethereum private key. You sign a
`ClobAuth` message containing your address, timestamp, nonce, and a fixed
attestation string. The CLOB server verifies the signature and returns:

```
{
  "apiKey": "<uuid>",
  "secret": "<base64-hmac-secret>",
  "passphrase": "<string>"
}
```

L1 is a one-time operation per API key. Credentials can be cached and reused.

### L2: Ongoing Request Authentication

Every authenticated request uses HMAC-SHA256 signing with the `secret` from L1.

Required headers for each authenticated request:
- `POLY_ADDRESS`: your wallet address
- `POLY_API_KEY`: the API key from L1
- `POLY_PASSPHRASE`: the passphrase from L1
- `POLY_SIGNATURE`: HMAC-SHA256(secret, timestamp + method + path + body)
- `POLY_TIMESTAMP`: current Unix timestamp

### Signature Types

| Type | Value | Use Case |
|------|-------|----------|
| EOA | 0 | MetaMask, hardware wallet, raw private key |
| Proxy | 1 | Magic/email wallets (funder auto-derived via CREATE2) |
| GnosisSafe | 2 | Browser extension proxy wallets |

### What We Need for Paper Trading

**Read-only market data: NO auth required.** Unauthenticated client covers:
- Market listing, search, metadata (Gamma API)
- Prices, spreads, midpoints, orderbook snapshots (CLOB API)
- WebSocket market data streams (WSS)
- Historical trades, leaderboards, positions (Data API)

**Live order submission (future live trading): L1 + L2 required.**
When we're ready to go from paper to live, we'll need a private key for signing.
The `polymarket-client-sdk` handles all of this via a type-state pattern:
`Client<Unauthenticated>` → `Client<Authenticated<Normal>>`.

---

## 3. WebSocket Channels

The WSS feed provides two channel types:

### Market Channel (public, no auth)

| Stream | Event Type | Data |
|--------|------------|------|
| `subscribe_orderbook()` | `book` | Full/delta orderbook: bids[], asks[], asset_id, hash |
| `subscribe_prices()` | `price_change` | Price changes with best_bid/ask, side, size |
| `subscribe_midpoints()` | (derived from book) | Calculated midpoint prices |
| `subscribe_last_trade_price()` | `last_trade_price` | Price, side, size, fee_rate_bps |
| `subscribe_tick_size_change()` | `tick_size_change` | Old/new tick size per asset |
| `subscribe_best_bid_ask()` | `best_bid_ask` | Best bid/ask (requires custom flag) |
| `subscribe_new_markets()` | `new_market` | New market creation events |
| `subscribe_market_resolutions()` | `market_resolved` | Market resolution events |

### User Channel (authenticated)

| Stream | Event Type | Data |
|--------|------------|------|
| `subscribe_orders()` | `order` | Order status updates for your orders |
| `subscribe_trades()` | `trade` | Trade execution notifications |

### Connection Management

- Heartbeat interval: 5s (PING)
- Heartbeat timeout: 15s (PONG deadline)
- Auto-reconnect: exponential backoff, 1s initial, 60s max, infinite retries
- Multiplexed subscriptions: single WS connection per channel type, ref-counted

### For Our Paper Trading System

We need only the **market channel** (unauthenticated):
- `subscribe_orderbook()` for live order book to simulate fills
- `subscribe_prices()` for price signals to feed the TPP model
- `subscribe_last_trade_price()` for trade flow (volume, timing)

One WSS connection handles all weather market subscriptions. The Rust SDK
handles reconnection and heartbeats automatically.

---

## 4. Rate Limits

All rate limits enforced via Cloudflare sliding window. Requests are throttled
(queued), not immediately rejected. HTTP 429 returned when exceeded.

### CLOB API (`clob.polymarket.com`)

| Endpoint Category | Burst (per 10s) | Sustained (per 10min) |
|-------------------|-----------------|----------------------|
| General | 9,000 | — |
| Market data (book, price, midpoint) | 1,500 | — |
| Batch market data | 500 | — |
| `POST /order` | 3,500 | 36,000 |
| `DELETE /order` | 3,000 | 30,000 |
| Batch order ops | 1,000 | 15,000 |
| `DELETE /cancel-all` | 250 | 6,000 |
| Balance GET | 200 | — |
| Balance UPDATE | 50 | — |
| Auth endpoints | 100 | — |

### Gamma API (`gamma-api.polymarket.com`)

| Endpoint | Burst (per 10s) |
|----------|-----------------|
| General | 4,000 |
| `/events` | 500 |
| `/markets` | 300 |

### Data API (`data-api.polymarket.com`)

| Endpoint | Burst (per 10s) |
|----------|-----------------|
| General | 1,000 |
| `/trades` | 200 |
| `/positions` | 150 |

### WSS

No explicit rate limit on incoming market data. Push-based, multiplexed
internally. Heartbeat: send PING every 10s, server responds PONG.

### Practical Implications

Our paper trading loop runs at 15-30s intervals. With WSS for market data,
we make zero REST calls in the hot path. Rate limits are irrelevant for our
use case. Even REST polling 10 markets at 15s intervals = 0.67 req/s, well
within the 150 req/s burst limit for market data.

---

## 5. Tick Rules

### Tick Sizes

Markets have per-asset minimum tick sizes:

| Tick Size | Decimal | Typical Use |
|-----------|---------|-------------|
| Tenth | 0.1 | Low-liquidity or wide markets |
| Hundredth | 0.01 | **Standard (most markets)** |
| Thousandth | 0.001 | Higher-precision markets |
| TenThousandth | 0.0001 | Maximum precision |

Tick size is queryable per-token via `GET /tick-size?token_id=<id>` or via the
`tick_size` field in the orderbook response. Tick size can change dynamically
(price crosses thresholds) — the WSS fires `tick_size_change` events.

### Lot Size

Maximum 2 decimal places for order `size` (number of contracts).
`LOT_SIZE_SCALE = 2` in the SDK.

### Price Range

- Prices: $0.01 to $0.99 (binary outcome markets)
- USDC denomination: 6 decimals (Polygon USDC)

### Order Types

| Type | Behavior |
|------|----------|
| GTC | Good 'til Cancelled — rests on book until filled or cancelled |
| FOK | Fill or Kill — immediate full fill or cancel entirely |
| GTD | Good 'til Date — rests on book until specified date |
| FAK | Fill and Kill — fill what you can immediately, cancel remainder |

For our maker strategy: **GTC** orders (rest on book, wait for taker flow).

---

## 6. Maker Rebate & Settlement

### Fee Structure (Weather Markets)

| Parameter | Value |
|-----------|-------|
| θ (theta) | 0.050 |
| Maker fee | **$0.00** (zero) |
| Taker fee | θ × C × p × (1 − p) |
| Maker rebate | 25% of taker fee pool, pro-rata |

Fee formula: `fee = 0.05 × contracts × price × (1 - price)`

At p=0.50, 100 contracts: taker fee = $1.25. At resolution (p→1.0 or 0.0):
taker fee → $0.00.

### Maker Rebate Mechanics

- Pool: 25% of taker fees collected in the market, daily
- Distribution: pro-rata by fee-equivalent volume
- `fee_equiv = θ × C × p × (1 − p)` (computed as if maker fill were a taker)
- Queryable via SDK: `client.rewards_percentages()`, `client.user_earning()`

**For cost modeling: omit the rebate.** It's pooled, unpredictable, and
conservative to ignore.

### Settlement (Resolution)

Binary markets resolve to exactly 0 or 1 via **UMA Optimistic Oracle**:
- Proposal (bond ~$750 USDC.e) → 2-hour challenge period → resolution
- Undisputed: ~2 hours. Disputed: 4-6 days (UMA voter vote).
- **YES tokens** redeem for $1.00 USDC.e; **NO tokens** = $0.00
- Resolution triggers `market_resolved` WSS event
- At resolution, taker exit fee → $0 (price at extreme)
- Redemption: call `redeemPositions` on CTF contract (burns tokens, returns collateral)
- **Holding rewards**: 4.00% annualized on total position value (hourly sample, daily payout)

**Neg risk markets**: some multi-outcome markets use a "negative risk" model
where buying YES on one outcome implicitly sells all others. Uses a distinct
exchange contract. The SDK exposes `neg_risk` per token via `client.neg_risk(token_id)`.

### Operational Notes

- **Matching engine restarts**: Weekly, Tuesdays 7:00 AM ET (~90s downtime).
  Returns HTTP 425 during restart — use exponential backoff.
- **Gasless**: Relayer pays gas for wallet deploy, approvals, CTF ops.
- **Sports markets**: 3-second matching delay for marketable orders.
- **Batch book**: up to 500 tokens per request.
- **Heartbeat for live trading**: must send `POST /heartbeats` every 10s
  (with 5s buffer). All open orders auto-cancelled on heartbeat lapse.

### For Paper Trading

We simulate maker entries (GTC limit orders) and track fills against live
orderbook data. Costs are:
- Entry: $0.00 (maker)
- Exit at resolution: $0.00 (price extreme)
- Exit early at p: θ × C × p × (1 − p)
- Adverse selection: measured from post-fill drift

---

## 7. rs-clob-client Assessment

### Crate: `polymarket-client-sdk` v0.4.4

| Attribute | Value |
|-----------|-------|
| Published | crates.io, MIT license |
| Downloads | ~87K total |
| MSRV | Rust 1.88.0 (edition 2024) |
| Last updated | 2026-04-06 |
| CI | Green, with benchmarks and coverage |
| Maintainer | Polymarket Engineering (official) |

### Feature Coverage

| Feature | What It Provides | Need for Paper Trading |
|---------|-----------------|----------------------|
| `clob` | REST client: markets, prices, orderbook, orders, trades, balances | **Required** |
| `ws` | WSS client: orderbook, prices, midpoints, user events | **Required** |
| `data` | Positions, trades, leaderboards, analytics | Useful |
| `gamma` | Market/event discovery and search | **Required** |
| `bridge` | Cross-chain deposits | Not needed |
| `rfq` | Request-for-quote | Not needed |
| `ctf` | Split/merge/redeem tokens | Not needed |
| `heartbeats` | Auto-heartbeat for order keepalive | Not needed (paper) |
| `tracing` | Structured logging | Recommended |

### Architecture Quality

- **Type-state auth**: `Client<Unauthenticated>` vs `Client<Authenticated<K>>`.
  Compile-time enforcement that you can't call trading endpoints without auth.
- **Zero-cost abstractions**: no dynamic dispatch in hot paths
- **Streaming pagination**: `stream_data()` for iterating through large result sets
- **Auto-reconnect WSS**: exponential backoff, heartbeats, multiplexed connections
- **Modern Ethereum**: uses `alloy` (not deprecated `ethers-rs`)

### Dependencies for Paper Trading

```toml
[dependencies]
polymarket-client-sdk = { version = "0.4", features = ["clob", "ws", "gamma", "data", "tracing"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Verdict

**Use it.** It's the official client, well-maintained, covers everything we need,
and the type-state pattern prevents auth mistakes at compile time. No need for
alternatives.

---

## 8. NautilusTrader Evaluation

NautilusTrader is a high-performance algorithmic trading platform (Rust core +
Python API) by Nautilus Systems. It provides a full trading framework:
event-driven architecture, backtesting, live execution, risk management.

### Polymarket Integration Status

**Production-grade, first-class adapter.** Not experimental or community-contributed.

- Rust crate: `nautilus-polymarket` (in `crates/adapters/polymarket/`)
- Python package: `nautilus_trader.adapters.polymarket`
- 63+ commits, last updated 2026-04-09 (daily active development)
- Shipping in official releases (v1.225.0+)

**What it provides:**

| Capability | Details |
|------------|---------|
| Data feeds | WSS streaming: orderbook, quotes (BBO), trade ticks, price changes, tick size changes, new markets |
| REST data | Orderbook snapshots, historical trades, market metadata |
| Market discovery | Instrument provider via Gamma API with slug/event/composite filters |
| Order execution | Full lifecycle: submit, cancel, cancel-all, batch cancel |
| Order types | MARKET, LIMIT; TIF: GTC, GTD, FOK, IOC |
| Fill tracking | Dust detection, position reconciliation from on-chain balances |
| EIP-712 signing | Native Rust (sub-millisecond), no Python dependency |
| Rate limiting | Built-in: 100 req/min public, 300/min auth, 60 orders/min |
| WSS connection mgmt | Auto-manages 200 instruments/connection (up to 500 configurable) |
| Backtesting | Full backtest support with historical Polymarket data |

### Sandbox (Paper Trading) Support

NautilusTrader has a **dedicated sandbox adapter** (`crates/adapters/sandbox/`):

- `SandboxExecutionClient`: uses `OrderMatchingEngine` to simulate execution
  against live market data
- Supports all order types, configurable fill/fee models, account balance tracking
- **Architecture pattern**: real data client (Polymarket) + sandbox execution client

```rust
// Conceptual setup:
let mut node = LiveNode::builder(trader_id, environment)?
    .add_data_client(None, Box::new(PolymarketDataClientFactory), Box::new(data_config))?
    .add_exec_client(Some("POLYMARKET"), Box::new(SandboxExecutionClientFactory::new()), Box::new(sandbox_config))?
    .build()?;
```

### NautilusTrader vs Custom Rust Loop: Trade-offs

| Factor | NautilusTrader | Custom Rust Loop |
|--------|---------------|------------------|
| **Time to first paper trade** | Faster — plumbing is done | Slower — must build feed, fill sim, position tracking |
| **Fill simulation fidelity** | High — `OrderMatchingEngine` is battle-tested | Must build and validate |
| **Complexity** | High — full framework, own event model, config | Low — just `polymarket-client-sdk` + our logic |
| **TPP integration** | Strategy must fit NautilusTrader's event model | Direct — call ONNX wherever we want |
| **Live trading flip** | Swap `SandboxExecClient` → `PolymarketExecClient` | Swap paper fills → real order API calls |
| **Dependencies** | Heavy — Cython builds, LGPL-v3 license | Light — only `polymarket-client-sdk` |
| **Learning curve** | Significant — must learn NautilusTrader patterns | Minimal — standard Rust async |
| **Maintenance** | We depend on upstream adapter updates | We own the code |

### Recommendation

**Build the custom Rust loop.** Despite NautilusTrader's impressive adapter:

1. Our signal pipeline (TPP ONNX + volume baseline) is specific and doesn't
   fit naturally into NautilusTrader's `Strategy` trait pattern.
2. The paper trading engine is simple — we need fill simulation against live
   book data, not a full matching engine with margin accounting.
3. LGPL-v3 license creates distribution concerns if we ever package this.
4. The custom loop is ~500 lines of focused code vs learning a framework.
5. For live trading flip, swapping `paper_fill()` → `client.post_order()` in
   our own code is trivial.

**However**, NautilusTrader remains a viable option if our requirements grow
(multi-market, risk limits, complex order routing). File this as an alternative
path for later convoys.

---

## 9. Recommended Architecture

### System Overview

```
┌──────────────────────────────────────────────────┐
│                  Paper Trading Engine (Rust)       │
│                                                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ WSS Feed │  │ Signal   │  │ Paper Execution│  │
│  │ (market  │→ │ Engine   │→ │ Engine         │  │
│  │  data)   │  │ (TPP +   │  │ (simulated     │  │
│  │          │  │  volume) │  │  fills + P&L)  │  │
│  └──────────┘  └──────────┘  └────────────────┘  │
│       ↑                             │             │
│       │                             ↓             │
│  ┌──────────┐              ┌────────────────┐    │
│  │ Gamma    │              │ State/Storage  │    │
│  │ (market  │              │ (positions,    │    │
│  │  disco)  │              │  trades, P&L)  │    │
│  └──────────┘              └────────────────┘    │
│                                     │             │
│                                     ↓             │
│                            ┌────────────────┐    │
│                            │ Viz Export     │    │
│                            │ (JSON → viz    │    │
│                            │  site)         │    │
│                            └────────────────┘    │
└──────────────────────────────────────────────────┘
```

### Components

**1. WSS Market Data Feed** (Convoy 2: pm-38v.2)
- Subscribe to weather market orderbooks and prices via `polymarket-client-sdk` WS
- Feed into signal engine as events
- No auth required

**2. Signal Engine** (Convoy 3: pm-38v.3)
- TPP model (ONNX runtime): 15-30s tick, predict timing intensity
- Volume baseline: 60-120s tick, rolling volume spike detector
- Both produce BUY-only signals (SELL signals filtered per recommendation)
- 5-minute drift horizon

**3. Paper Execution Engine** (Convoy 4: pm-38v.4)
- Receives BUY signals from signal engine
- Simulates GTC limit order placement at current best ask (or midpoint - spread)
- Tracks fill simulation using live orderbook data:
  - Fill if taker volume crosses our price level within fill window
  - Apply cost model: $0 entry (maker), taker exit fee, adverse selection
- Position tracking: open/closed positions, P&L per trade, cumulative P&L
- No real orders submitted

**4. State & Monitoring** (Convoy 5: pm-38v.5)
- JSON state export for pmgetrich-viz.fly.dev
- Metrics: hit rate, Sharpe, cumulative P&L, trades/day
- Alert if signals stop or data feed disconnects

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Custom Rust loop, not NautilusTrader | Simplicity; NautilusTrader is overkill |
| `polymarket-client-sdk` as only dependency | Official, well-typed, covers all needs |
| WSS for data, not REST polling | Lower latency, no rate limit concerns |
| BUY-only | SELL signals adversely selected (per backtest) |
| Maker simulation only | Zero entry fees, conservative fill model |
| ONNX for TPP inference | Avoids Python/PyTorch in the fast loop |
| 5-minute drift horizon | Best Sharpe observed in backtest |

### Dependency Stack

```toml
[dependencies]
polymarket-client-sdk = { version = "0.4", features = ["clob", "ws", "gamma", "data", "tracing"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros", "signal"] }
ort = "2"                    # ONNX Runtime for TPP inference
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
chrono = { version = "0.4", features = ["serde"] }
rust_decimal = "1"
anyhow = "1"
```

### Crate Structure (proposed)

```
pmgetrich-paper/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point, tokio runtime
│   ├── config.rs            # Market selection, signal params
│   ├── feed/
│   │   ├── mod.rs
│   │   └── ws.rs            # WSS subscription manager
│   ├── signal/
│   │   ├── mod.rs
│   │   ├── tpp.rs           # TPP ONNX inference
│   │   └── volume.rs        # Volume baseline
│   ├── execution/
│   │   ├── mod.rs
│   │   ├── paper.rs         # Paper order/fill simulation
│   │   └── position.rs      # Position tracking
│   ├── cost.rs              # Fee/fill cost model
│   └── export.rs            # JSON export for viz
├── models/
│   └── tpp.onnx             # Exported TPP model
└── data/
    └── paper_trades.json    # Output
```

---

## 10. Open Questions for Review

1. **Market selection**: Which weather markets to subscribe to? All active, or
   a curated list? The Gamma API can filter by category/tag.

2. **ONNX model export**: The TPP model is currently PyTorch. Need to verify
   ONNX export works correctly for the Neural Hawkes architecture. If ONNX
   doesn't support the custom attention layers, we may need to call Python
   as a subprocess (slower but simpler).

3. **Fill simulation fidelity**: Current cost model assumes queue_position=1.0
   (worst case). With live orderbook data, we can compute actual depth ahead
   of our simulated order for better fill probability estimates.

4. **Dual strategy comparison**: Run TPP and volume baseline side-by-side in
   paper trading, or just TPP? Running both adds minimal complexity (both
   read the same data) and provides live comparison data.

5. **Data retention**: How long to keep paper trades? 7 days (per Convoy 6
   assessment window), or indefinitely for ongoing analysis?

6. **Wallet setup**: Even for read-only, the Gamma API and Data API queries
   for positions require a wallet address. Do we have a Polymarket wallet
   already, or need to create one?

---

## 11. Contract Addresses (Polygon, Chain 137)

| Contract | Address |
|----------|---------|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` |
| Neg Risk CTF Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Conditional Tokens (CTF) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |
| USDC.e (6 decimals) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |

Not needed for paper trading, but required for live order signing (the exchange
address is part of the EIP-712 domain).

---

## Sources

- Polymarket CLOB API: docs.polymarket.com
- rs-clob-client: github.com/Polymarket/rs-clob-client (v0.4.4)
- polymarket-client-sdk: crates.io/crates/polymarket-client-sdk
- NautilusTrader: github.com/nautechsystems/nautilus_trader
- Fee schedule: polymarketexchange.com/fees-hours.html
- Prior work: cost_model.md, recommendation.md, edge_assessment.md (this repo)
