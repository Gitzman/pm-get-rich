"""Generate trading signals from pre-trained cross-market transformer TPP.

INFERENCE ONLY — loads existing model, does NOT train anything.

For each held-out event (end_date >= 2026-03-25), slides a context window
through the event sequence and at each step:
1. Gets model's prediction for next event (bucket position, time-to-next)
2. If predicted time-to-next < Δt AND predicted bucket shows strong activity,
   emits a signal with the predicted bucket and direction.

Output: data/signals/signals.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition (must match training exactly)
# ---------------------------------------------------------------------------

def sinusoidal_time_encoding(dt: torch.Tensor, dim: int) -> torch.Tensor:
    log_dt = torch.log1p(dt).unsqueeze(-1)
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=dt.device, dtype=dt.dtype)
        * (-math.log(10000.0) / dim)
    )
    args = log_dt * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CrossMarketTPP(nn.Module):
    def __init__(self, n_wallets, n_cities, d_model=128, n_heads=4,
                 n_layers=4, d_ff=256, dropout=0.1, wallet_dim=64,
                 city_dim=16, side_dim=8, time_dim=32):
        super().__init__()
        self.d_model = d_model
        self.n_wallets = n_wallets
        self.wallet_dim = wallet_dim
        self.wallet_embed = nn.Embedding(n_wallets, wallet_dim)
        self.city_embed = nn.Embedding(n_cities, city_dim)
        self.side_embed = nn.Embedding(2, side_dim)
        self.bucket_proj = nn.Linear(1, 16)
        self.price_proj = nn.Linear(1, 8)
        self.context_proj = nn.Linear(2, 8)
        self.time_dim = time_dim
        feat_dim = wallet_dim + city_dim + side_dim + 16 + 8 + 8 + time_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.wallet_head = nn.Linear(d_model, n_wallets)
        self.bucket_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.time_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, wallet_idx, city_idx, side_idx, bucket_pos, price,
                time_delta, hours_to_res, n_buckets):
        B, L = wallet_idx.shape
        w = self.wallet_embed(wallet_idx)
        c = self.city_embed(city_idx)
        s = self.side_embed(side_idx)
        b = self.bucket_proj(bucket_pos.unsqueeze(-1))
        p = self.price_proj(price.unsqueeze(-1))
        ctx = torch.stack([hours_to_res / 200.0, n_buckets / 11.0], dim=-1)
        ctx_enc = self.context_proj(ctx)
        t = sinusoidal_time_encoding(time_delta, self.time_dim)
        features = torch.cat([w, c, s, b, p, ctx_enc, t], dim=-1)
        x = self.input_proj(features)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        wallet_logits = self.wallet_head(h)
        bucket_pred = self.bucket_head(h).squeeze(-1)
        time_params = self.time_head(h)
        return wallet_logits, bucket_pred, time_params[..., 0], time_params[..., 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_bucket_positions(df: pl.DataFrame, events_dir: Path) -> pl.DataFrame:
    """Add bucket_position column (0=floor, 1=ceiling) from suit labels."""
    eid = df["event_id"][0]
    meta_path = events_dir / str(eid) / "_meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        buckets = sorted(meta.get("buckets", []))
    else:
        buckets = sorted(df["suit"].unique().to_list())
    n = max(len(buckets), 1)
    bucket_map = {b: i / max(n - 1, 1) for i, b in enumerate(buckets)}
    positions = [bucket_map.get(s, 0.5) for s in df["suit"].to_list()]
    return df.with_columns(pl.Series("bucket_position", positions, dtype=pl.Float32))


def load_model(model_dir: Path, device: torch.device):
    """Load the pre-trained cross-market transformer."""
    vocab_path = model_dir / "vocab.json"
    model_path = model_dir / "model.pt"

    vocab = json.load(open(vocab_path))
    wallet_to_idx = {w["address"]: w["idx"] for w in vocab.get("wallets", [])}
    city_to_idx = vocab.get("city_to_idx", {})
    n_wallets = vocab.get("n_wallets", len(wallet_to_idx) + 1)
    n_cities = vocab.get("n_cities", len(city_to_idx))
    unk_wallet_idx = n_wallets - 1

    model = CrossMarketTPP(n_wallets=n_wallets, n_cities=n_cities)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, wallet_to_idx, city_to_idx, unk_wallet_idx


def prepare_event(df: pl.DataFrame, meta: dict, wallet_to_idx: dict,
                  city_to_idx: dict, unk_wallet_idx: int, events_dir: Path):
    """Prepare feature arrays from an event dataframe."""
    df = df.sort("seq")
    df = compute_bucket_positions(df, events_dir)

    actors = df["actor"].to_list()
    timestamps = df["timestamp_ms"].to_numpy().astype(np.float64)
    bucket_positions = df["bucket_position"].to_numpy().astype(np.float32)
    prices = df["price"].to_numpy().astype(np.float32)
    sides = df["side"].to_list()
    suits = df["suit"].to_list()

    wallet_indices = np.array([wallet_to_idx.get(a, unk_wallet_idx) for a in actors], dtype=np.int64)
    city = meta.get("city", "unknown")
    city_idx_val = city_to_idx.get(city, 0)
    city_indices = np.full(len(actors), city_idx_val, dtype=np.int64)
    side_indices = np.array([0 if s == "BUY" else 1 for s in sides], dtype=np.int64)

    time_hours = (timestamps - timestamps[0]) / 3_600_000.0
    time_deltas = np.zeros_like(time_hours)
    time_deltas[1:] = np.diff(time_hours)
    time_deltas = np.clip(time_deltas, 0, None)

    end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
    hours_to_res = np.zeros(len(actors), dtype=np.float32)
    if end_s > 0:
        for i, ts in enumerate(timestamps):
            hours_to_res[i] = max(0, (end_s * 1000 - ts) / 3_600_000.0)

    n_buckets_val = meta.get("n_buckets", 11)
    n_buckets_arr = np.full(len(actors), float(n_buckets_val), dtype=np.float32)

    return {
        "wallet_indices": wallet_indices,
        "city_indices": city_indices,
        "side_indices": side_indices,
        "bucket_positions": bucket_positions,
        "prices": prices,
        "time_deltas": time_deltas,
        "hours_to_res": hours_to_res,
        "n_buckets": n_buckets_arr,
        "timestamps": timestamps,
        "suits": suits,
        "sides": sides,
    }


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals_for_event(
    model: CrossMarketTPP,
    event_id: str,
    events_dir: Path,
    wallet_to_idx: dict,
    city_to_idx: dict,
    unk_wallet_idx: int,
    device: torch.device,
    context_len: int = 512,
    dt_values: list[int] = [15, 30, 60, 120],
    threshold_pcts: list[int] = [1, 5, 10],
) -> list[dict]:
    """Generate signals for one event using the pre-trained model."""
    parquet_path = events_dir / event_id / "events.parquet"
    meta_path = events_dir / event_id / "_meta.json"
    if not parquet_path.exists():
        return []

    df = pl.read_parquet(parquet_path).sort("seq")
    if df.height < context_len + 10:
        return []

    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    feats = prepare_event(df, meta, wallet_to_idx, city_to_idx, unk_wallet_idx, events_dir)

    N = len(feats["wallet_indices"])
    # Use last 20% as signal generation window (same split as eval)
    split = int(0.8 * N)
    if N - split < context_len + 1:
        return []

    # Collect predictions at each step in the test window
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(split, N - 1, max(1, context_len // 4)):  # overlapping windows
            start = max(0, i - context_len)
            end = i
            L = end - start
            if L < 10:
                continue

            inp = {
                "wallet_idx": torch.tensor(feats["wallet_indices"][start:end][np.newaxis], dtype=torch.long, device=device),
                "city_idx": torch.tensor(feats["city_indices"][start:end][np.newaxis], dtype=torch.long, device=device),
                "side_idx": torch.tensor(feats["side_indices"][start:end][np.newaxis], dtype=torch.long, device=device),
                "bucket_pos": torch.tensor(feats["bucket_positions"][start:end][np.newaxis], dtype=torch.float32, device=device),
                "price": torch.tensor(feats["prices"][start:end][np.newaxis], dtype=torch.float32, device=device),
                "time_delta": torch.tensor(feats["time_deltas"][start:end][np.newaxis], dtype=torch.float32, device=device),
                "hours_to_res": torch.tensor(feats["hours_to_res"][start:end][np.newaxis], dtype=torch.float32, device=device),
                "n_buckets": torch.tensor(feats["n_buckets"][start:end][np.newaxis], dtype=torch.float32, device=device),
            }

            wl, bp, tm, tls = model(**inp)

            # Last position prediction = next event prediction
            pred_bucket_pos = torch.sigmoid(bp[0, -1]).item()
            pred_time_mu = tm[0, -1].item()
            pred_time_log_sigma = tls[0, -1].item()
            pred_time_hours = math.exp(pred_time_mu)  # log-normal mean
            pred_time_seconds = pred_time_hours * 3600

            # Actual next event info
            actual_suit = feats["suits"][i] if i < N else None
            actual_price = feats["prices"][i] if i < N else None
            actual_side = feats["sides"][i] if i < N else None
            actual_bucket_pos = feats["bucket_positions"][i] if i < N else None
            signal_ts = int(feats["timestamps"][i - 1])  # timestamp of last observed event

            # Price at signal time (last observed)
            current_price = feats["prices"][i - 1]
            current_suit = feats["suits"][i - 1]

            # Look ahead: what happened in the next Δt seconds?
            for dt_s in dt_values:
                dt_ms = dt_s * 1000
                future_mask = (feats["timestamps"][i:] - feats["timestamps"][i - 1]) <= dt_ms
                n_future = int(future_mask.sum())

                if n_future > 0:
                    future_prices = feats["prices"][i:i + n_future]
                    future_suits = feats["suits"][i:i + n_future]
                    price_at_end = future_prices[-1]
                    price_change = float(price_at_end - current_price)
                    max_price = float(future_prices.max())
                    min_price = float(future_prices.min())
                else:
                    price_change = 0.0
                    max_price = float(current_price)
                    min_price = float(current_price)

                predictions.append({
                    "event_id": event_id,
                    "city": meta.get("city", "unknown"),
                    "timestamp_ms": signal_ts,
                    "dt_seconds": dt_s,
                    "pred_bucket_pos": pred_bucket_pos,
                    "pred_time_seconds": pred_time_seconds,
                    "current_price": float(current_price),
                    "current_suit": current_suit,
                    "actual_next_suit": actual_suit,
                    "actual_next_price": float(actual_price) if actual_price is not None else None,
                    "actual_next_side": actual_side,
                    "n_future_events": n_future,
                    "price_change": price_change,
                    "max_price_in_window": max_price,
                    "min_price_in_window": min_price,
                    "bucket_pred_confidence": abs(pred_bucket_pos - 0.5) * 2,  # 0=uncertain, 1=confident
                })

    if not predictions:
        return []

    # Apply thresholds: rank by bucket_pred_confidence, keep top N%
    signals = []
    for dt_s in dt_values:
        dt_preds = [p for p in predictions if p["dt_seconds"] == dt_s]
        if not dt_preds:
            continue
        confidences = sorted([p["bucket_pred_confidence"] for p in dt_preds], reverse=True)
        for thr_pct in threshold_pcts:
            cutoff_idx = max(1, int(len(confidences) * thr_pct / 100))
            cutoff_val = confidences[min(cutoff_idx, len(confidences) - 1)]
            for p in dt_preds:
                if p["bucket_pred_confidence"] >= cutoff_val:
                    signals.append({**p, "threshold_pct": thr_pct})

    return signals


def main():
    parser = argparse.ArgumentParser(description="Generate signals from pre-trained TPP (INFERENCE ONLY)")
    parser.add_argument("--model-dir", type=Path, default=Path("data/reports/fullscale/pre_march25"))
    parser.add_argument("--vocab-dir", type=Path, default=Path("data/reports/generalization/cross_market"))
    parser.add_argument("--events-dir", type=Path, default=Path("data/events"))
    parser.add_argument("--out", type=Path, default=Path("data/signals/signals.parquet"))
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import datetime
    CUTOFF_EPOCH = datetime.datetime(2026, 3, 25, tzinfo=datetime.timezone.utc).timestamp()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pre-trained model
    print("Loading pre-trained model...")
    model, wallet_to_idx, city_to_idx, unk_wallet_idx = load_model(args.vocab_dir, device)
    # Load model weights from fullscale training
    state = torch.load(args.model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Wallets: {len(wallet_to_idx):,}")
    print(f"  Cities: {len(city_to_idx)}")

    # Find held-out events
    print("\nFinding held-out events...")
    holdout_eids = []
    for p in sorted(args.events_dir.iterdir()):
        meta_path = p / "_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
        if end_s >= CUTOFF_EPOCH:
            holdout_eids.append(p.name)
    print(f"  Found {len(holdout_eids)} held-out events")

    # Generate signals
    print(f"\nGenerating signals...")
    all_signals = []
    t0 = time.time()
    for i, eid in enumerate(holdout_eids):
        sigs = generate_signals_for_event(
            model, eid, args.events_dir,
            wallet_to_idx, city_to_idx, unk_wallet_idx,
            device, context_len=args.context_len,
        )
        all_signals.extend(sigs)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(holdout_eids)}] processed, {len(all_signals):,} signals so far ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_signals):,} signals from {len(holdout_eids)} events in {elapsed:.1f}s")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(all_signals)
    df.write_parquet(args.out)
    print(f"Saved: {args.out} ({len(df):,} rows)")

    # Quick summary
    print(f"\n=== Signal Summary ===")
    print(f"Events: {df['event_id'].n_unique()}")
    print(f"Cities: {df['city'].n_unique()}")
    for dt in [15, 30, 60, 120]:
        for thr in [1, 5, 10]:
            subset = df.filter((pl.col("dt_seconds") == dt) & (pl.col("threshold_pct") == thr))
            if len(subset) > 0:
                nonzero = (subset["price_change"].abs() > 0.001).sum()
                print(f"  dt={dt:>3}s thr={thr:>2}%: {len(subset):>6,} signals, "
                      f"{nonzero}/{len(subset)} with price move ({nonzero/len(subset):.0%})")


if __name__ == "__main__":
    main()
