"""Cross-market Transformer TPP with wallet embeddings.

Trains ONE neural temporal point process across all normalized markets.
Uses shared wallet embeddings, relative bucket positions, city embeddings,
and transformer attention to model cross-market trading patterns.

Key deliverables:
- Wallet embedding space analysis (t-SNE clustering)
- Attention-based influence detection
- Cross-market generalization metrics
- Comparison to per-market classical Hawkes LL

Usage:
    uv run python scripts/fit_cross_market_neural.py [--gpu 0] [--epochs 200]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def parse_suit_sort_key(suit: str) -> float:
    """Extract a sortable numeric value from a suit label.

    Handles formats: '<=53F', '>=72F', '54-55F', '10C', '<1.20', '>1.34', '1.20-1.24'
    """
    s = suit.strip()
    # Extract all numeric parts (including decimals and negatives)
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return 0.0
    if s.startswith("<=") or s.startswith("<"):
        return float(nums[0]) - 0.5  # sort below the lowest explicit bucket
    if s.startswith(">=") or s.startswith(">"):
        return float(nums[0]) + 0.5  # sort above the highest explicit bucket
    # Range like '54-55' or '1.20-1.24': use first number
    return float(nums[0])


def compute_bucket_positions(
    df: pl.DataFrame, meta_dir: Path
) -> pl.DataFrame:
    """Add bucket_position column (0=floor, 1=ceiling) to events dataframe."""
    positions = []
    for event_id in df["event_id"].unique().sort().to_list():
        meta_path = meta_dir / event_id / "_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            suits = meta.get("bucket_labels", [])
        else:
            # Fallback: get from data
            suits = (
                df.filter(pl.col("event_id") == event_id)["suit"]
                .unique()
                .sort()
                .to_list()
            )

        # Sort suits by temperature value
        sorted_suits = sorted(suits, key=parse_suit_sort_key)
        n = len(sorted_suits)
        suit_to_pos = {}
        for i, s in enumerate(sorted_suits):
            suit_to_pos[s] = i / max(n - 1, 1)

        positions.append((event_id, suit_to_pos))

    # Build mapping as a dictionary: (event_id, suit) -> position
    pos_map: dict[tuple[str, str], float] = {}
    for event_id, suit_map in positions:
        for suit, pos in suit_map.items():
            pos_map[(event_id, suit)] = pos

    # Add column
    bucket_pos = [
        pos_map.get((eid, suit), 0.5)
        for eid, suit in zip(
            df["event_id"].to_list(), df["suit"].to_list()
        )
    ]
    return df.with_columns(pl.Series("bucket_position", bucket_pos))


def load_cross_market_data(
    events_dir: Path,
    top_k_wallets: int = 500,
    min_wallet_events: int = 5,
) -> dict:
    """Load all events, build vocabularies, compute features.

    Returns dict with all arrays and metadata needed for training.
    """
    # Load consolidated events
    all_path = events_dir / "all_events.parquet"
    df = pl.read_parquet(all_path)
    print(f"  Loaded {df.height:,} events from {df['event_id'].n_unique()} markets")

    # Compute bucket positions
    df = compute_bucket_positions(df, events_dir)

    # Sort globally by timestamp
    df = df.sort("timestamp_ms", "event_id", "seq")

    # Build wallet vocabulary (top K by frequency)
    wallet_counts = (
        df.group_by("actor")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_wallet_events)
        .sort("n", descending=True)
        .head(top_k_wallets)
    )
    wallet_list = wallet_counts["actor"].to_list()
    wallet_to_idx = {w: i for i, w in enumerate(wallet_list)}
    unk_wallet_idx = len(wallet_list)
    n_wallets = len(wallet_list) + 1  # +1 for UNK
    print(f"  Wallet vocabulary: {len(wallet_list)} wallets + UNK (min {min_wallet_events} events)")

    # Build city vocabulary
    city_list: list[str] = []
    city_to_idx: dict[str, int] = {}
    for event_id in sorted(df["event_id"].unique().to_list()):
        meta_path = events_dir / event_id / "_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            city = meta.get("city", "unknown")
        else:
            city = "unknown"
        if city not in city_to_idx:
            city_to_idx[city] = len(city_list)
            city_list.append(city)
    n_cities = len(city_list)
    print(f"  City vocabulary: {n_cities} cities: {city_list}")

    # Build event_id -> city mapping
    event_city: dict[str, str] = {}
    event_meta: dict[str, dict] = {}
    for event_id in sorted(df["event_id"].unique().to_list()):
        meta_path = events_dir / event_id / "_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            event_city[event_id] = meta.get("city", "unknown")
            event_meta[event_id] = meta
        else:
            event_city[event_id] = "unknown"
            event_meta[event_id] = {}

    # Build arrays
    actors = df["actor"].to_list()
    event_ids = df["event_id"].to_list()
    timestamps = df["timestamp_ms"].to_numpy().astype(np.float64)
    bucket_positions = df["bucket_position"].to_numpy().astype(np.float32)
    prices = df["price"].to_numpy().astype(np.float32)
    sides = df["side"].to_list()

    wallet_indices = np.array(
        [wallet_to_idx.get(a, unk_wallet_idx) for a in actors], dtype=np.int64
    )
    city_indices = np.array(
        [city_to_idx[event_city[eid]] for eid in event_ids], dtype=np.int64
    )
    side_indices = np.array(
        [0 if s == "BUY" else 1 for s in sides], dtype=np.int64
    )

    # Compute time deltas in hours
    time_hours = (timestamps - timestamps[0]) / 3_600_000.0
    time_deltas = np.zeros_like(time_hours)
    time_deltas[1:] = np.diff(time_hours)
    time_deltas = np.clip(time_deltas, 0, None)

    # Compute hours to resolution for each event
    hours_to_resolution = np.zeros(len(event_ids), dtype=np.float32)
    for i, (eid, ts) in enumerate(zip(event_ids, timestamps)):
        meta = event_meta.get(eid, {})
        end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
        if end_s > 0:
            hours_to_resolution[i] = max(0, (end_s * 1000 - ts) / 3_600_000.0)

    # N_buckets per event
    n_buckets = np.array(
        [event_meta.get(eid, {}).get("n_buckets", 11) for eid in event_ids],
        dtype=np.float32,
    )

    # Chronological 80/20 split
    N = len(wallet_indices)
    split_idx = int(0.8 * N)
    print(f"  Total events: {N:,}")
    print(f"  Train: {split_idx:,} ({100*split_idx/N:.1f}%)")
    print(f"  Test: {N - split_idx:,} ({100*(N-split_idx)/N:.1f}%)")

    # Per-market event id mapping for evaluation
    market_event_ids = np.array(event_ids)

    return {
        "wallet_indices": wallet_indices,
        "city_indices": city_indices,
        "side_indices": side_indices,
        "bucket_positions": bucket_positions,
        "prices": prices,
        "time_deltas": time_deltas,
        "hours_to_resolution": hours_to_resolution,
        "n_buckets": n_buckets,
        "split_idx": split_idx,
        "n_wallets": n_wallets,
        "n_cities": n_cities,
        "wallet_list": wallet_list,
        "wallet_to_idx": wallet_to_idx,
        "city_list": city_list,
        "city_to_idx": city_to_idx,
        "event_ids": market_event_ids,
        "event_meta": event_meta,
        "N": N,
    }


# ---------------------------------------------------------------------------
# Temporal encoding
# ---------------------------------------------------------------------------


def sinusoidal_time_encoding(dt: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of time deltas. Input (*, ), output (*, dim)."""
    # Use log(1 + dt) for numerical stability with large ranges
    log_dt = torch.log1p(dt).unsqueeze(-1)  # (*, 1)
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=dt.device, dtype=dt.dtype)
        * (-math.log(10000.0) / dim)
    )  # (dim/2,)
    args = log_dt * freqs  # (*, dim/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (*, dim)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CrossMarketTPP(nn.Module):
    """Transformer-based temporal point process for cross-market prediction.

    Processes interleaved events from multiple markets, using shared wallet
    embeddings and relative bucket positions for cross-market generalization.
    """

    def __init__(
        self,
        n_wallets: int,
        n_cities: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        wallet_dim: int = 64,
        city_dim: int = 16,
        side_dim: int = 8,
        time_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_wallets = n_wallets
        self.wallet_dim = wallet_dim

        # Entity embeddings
        self.wallet_embed = nn.Embedding(n_wallets, wallet_dim)
        self.city_embed = nn.Embedding(n_cities, city_dim)
        self.side_embed = nn.Embedding(2, side_dim)

        # Continuous feature projections
        self.bucket_proj = nn.Linear(1, 16)
        self.price_proj = nn.Linear(1, 8)
        self.context_proj = nn.Linear(2, 8)  # hours_to_resolution, n_buckets

        # Temporal encoding
        self.time_dim = time_dim

        # Input projection: combine all features into d_model
        feat_dim = wallet_dim + city_dim + side_dim + 16 + 8 + 8 + time_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output heads (predict NEXT event properties from current hidden state)
        self.wallet_head = nn.Linear(d_model, n_wallets)
        self.bucket_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # mu, log_sigma for log-normal
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize embeddings with smaller values
        nn.init.normal_(self.wallet_embed.weight, std=0.02)
        nn.init.normal_(self.city_embed.weight, std=0.02)
        nn.init.normal_(self.side_embed.weight, std=0.02)

    def _encode_events(
        self,
        wallet_idx: torch.Tensor,   # (B, L)
        city_idx: torch.Tensor,     # (B, L)
        side_idx: torch.Tensor,     # (B, L)
        bucket_pos: torch.Tensor,   # (B, L)
        price: torch.Tensor,        # (B, L)
        time_delta: torch.Tensor,   # (B, L)
        hours_to_res: torch.Tensor, # (B, L)
        n_buckets: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:
        """Encode event features into d_model vectors. Returns (B, L, d_model)."""
        w_emb = self.wallet_embed(wallet_idx)           # (B, L, wallet_dim)
        c_emb = self.city_embed(city_idx)               # (B, L, city_dim)
        s_emb = self.side_embed(side_idx)               # (B, L, side_dim)

        b_enc = self.bucket_proj(bucket_pos.unsqueeze(-1))  # (B, L, 16)
        p_enc = self.price_proj(price.unsqueeze(-1))        # (B, L, 8)

        # Context features (normalized)
        ctx = torch.stack([
            hours_to_res / 200.0,  # normalize ~0-1
            n_buckets / 11.0,      # normalize ~0-1
        ], dim=-1)
        ctx_enc = self.context_proj(ctx)  # (B, L, 8)

        # Temporal encoding
        t_enc = sinusoidal_time_encoding(time_delta, self.time_dim)  # (B, L, time_dim)

        # Concatenate all features
        features = torch.cat([w_emb, c_emb, s_emb, b_enc, p_enc, ctx_enc, t_enc], dim=-1)
        return self.input_proj(features)  # (B, L, d_model)

    def forward(
        self,
        wallet_idx: torch.Tensor,
        city_idx: torch.Tensor,
        side_idx: torch.Tensor,
        bucket_pos: torch.Tensor,
        price: torch.Tensor,
        time_delta: torch.Tensor,
        hours_to_res: torch.Tensor,
        n_buckets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (wallet_logits, bucket_pred, time_mu, time_log_sigma).

        All outputs are (B, L, ...) and represent predictions for the NEXT event
        given history up to and including position i.
        """
        B, L = wallet_idx.shape

        # Encode events
        x = self._encode_events(
            wallet_idx, city_idx, side_idx, bucket_pos, price,
            time_delta, hours_to_res, n_buckets,
        )  # (B, L, d_model)

        # Causal mask: position i can only attend to positions <= i
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)

        # Transformer forward
        h = self.transformer(x, mask=mask, is_causal=True)  # (B, L, d_model)

        # Output heads
        wallet_logits = self.wallet_head(h)       # (B, L, n_wallets)
        bucket_pred = self.bucket_head(h).squeeze(-1)  # (B, L)
        time_params = self.time_head(h)           # (B, L, 2)
        time_mu = time_params[..., 0]             # (B, L)
        time_log_sigma = time_params[..., 1]      # (B, L)

        return wallet_logits, bucket_pred, time_mu, time_log_sigma

    def get_attention_weights(
        self,
        wallet_idx: torch.Tensor,
        city_idx: torch.Tensor,
        side_idx: torch.Tensor,
        bucket_pos: torch.Tensor,
        price: torch.Tensor,
        time_delta: torch.Tensor,
        hours_to_res: torch.Tensor,
        n_buckets: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract attention weights from all layers. Returns list of (B, H, L, L)."""
        B, L = wallet_idx.shape
        x = self._encode_events(
            wallet_idx, city_idx, side_idx, bucket_pos, price,
            time_delta, hours_to_res, n_buckets,
        )

        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        attn_weights = []

        # Manual forward through transformer layers to capture attention
        for layer in self.transformer.layers:
            # Pre-norm
            x_norm = layer.norm1(x)
            # Self-attention with weight output
            attn_out, weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=mask, is_causal=True,
                need_weights=True, average_attn_weights=False,
            )
            attn_weights.append(weights.detach())
            x = x + attn_out
            # FFN
            x = x + layer._ff_block(layer.norm2(x))

        return attn_weights


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(
    wallet_logits: torch.Tensor,    # (B, L, n_wallets)
    bucket_pred: torch.Tensor,      # (B, L)
    time_mu: torch.Tensor,          # (B, L)
    time_log_sigma: torch.Tensor,   # (B, L)
    target_wallet: torch.Tensor,    # (B, L)
    target_bucket: torch.Tensor,    # (B, L)
    target_time: torch.Tensor,      # (B, L) - time delta to next event
    mask: torch.Tensor | None = None,  # (B, L) valid positions
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute decomposed NLL loss.

    Total NLL = -log P(wallet) - log P(bucket) - log P(time)
    """
    B, L = target_wallet.shape

    # Wallet cross-entropy
    wallet_loss = F.cross_entropy(
        wallet_logits.reshape(-1, wallet_logits.size(-1)),
        target_wallet.reshape(-1),
        reduction="none",
    ).reshape(B, L)

    # Bucket position MSE (in [0,1])
    bucket_loss = F.mse_loss(
        torch.sigmoid(bucket_pred), target_bucket, reduction="none"
    )

    # Time: log-normal NLL
    # log P(dt | mu, sigma) = -log(dt) - log(sigma) - 0.5*((log(dt) - mu)/sigma)^2 - 0.5*log(2*pi)
    log_sigma = time_log_sigma.clamp(-5, 5)
    sigma = torch.exp(log_sigma)
    log_dt = torch.log(target_time.clamp(min=1e-8))
    time_loss = (
        log_dt + log_sigma + 0.5 * ((log_dt - time_mu) / sigma.clamp(min=1e-6)) ** 2
        + 0.5 * math.log(2 * math.pi)
    )

    if mask is not None:
        wallet_loss = wallet_loss * mask
        bucket_loss = bucket_loss * mask
        time_loss = time_loss * mask
        n_valid = mask.sum().clamp(min=1)
    else:
        n_valid = torch.tensor(B * L, dtype=torch.float32, device=wallet_loss.device)

    wallet_mean = wallet_loss.sum() / n_valid
    bucket_mean = bucket_loss.sum() / n_valid
    time_mean = time_loss.sum() / n_valid

    total = wallet_mean + bucket_mean + time_mean

    metrics = {
        "wallet_nll": float(wallet_mean),
        "bucket_mse": float(bucket_mean),
        "time_nll": float(time_mean),
        "total_nll": float(total),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Dataset and batching
# ---------------------------------------------------------------------------


class EventWindowDataset:
    """Creates non-overlapping windows from the global event sequence."""

    def __init__(
        self,
        data: dict,
        start_idx: int,
        end_idx: int,
        context_len: int = 512,
    ):
        self.context_len = context_len
        self.data = data

        # Create window start indices (non-overlapping, need context_len+1 for targets)
        self.windows = []
        i = start_idx
        while i + context_len + 1 <= end_idx:
            self.windows.append(i)
            i += context_len

    def __len__(self) -> int:
        return len(self.windows)

    def get_batch(
        self, indices: list[int], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get a batch of windows. Returns (inputs, targets)."""
        L = self.context_len
        batch_w, batch_c, batch_s = [], [], []
        batch_bp, batch_p, batch_td = [], [], []
        batch_hr, batch_nb = [], []
        tgt_w, tgt_bp, tgt_td = [], [], []

        for idx in indices:
            start = self.windows[idx]
            # Input: positions [start, start+L)
            # Target: positions [start+1, start+L+1) (predict next event)
            batch_w.append(self.data["wallet_indices"][start : start + L])
            batch_c.append(self.data["city_indices"][start : start + L])
            batch_s.append(self.data["side_indices"][start : start + L])
            batch_bp.append(self.data["bucket_positions"][start : start + L])
            batch_p.append(self.data["prices"][start : start + L])
            batch_td.append(self.data["time_deltas"][start : start + L])
            batch_hr.append(self.data["hours_to_resolution"][start : start + L])
            batch_nb.append(self.data["n_buckets"][start : start + L])

            tgt_w.append(self.data["wallet_indices"][start + 1 : start + L + 1])
            tgt_bp.append(self.data["bucket_positions"][start + 1 : start + L + 1])
            tgt_td.append(self.data["time_deltas"][start + 1 : start + L + 1])

        inputs = {
            "wallet_idx": torch.tensor(np.stack(batch_w), dtype=torch.long, device=device),
            "city_idx": torch.tensor(np.stack(batch_c), dtype=torch.long, device=device),
            "side_idx": torch.tensor(np.stack(batch_s), dtype=torch.long, device=device),
            "bucket_pos": torch.tensor(np.stack(batch_bp), dtype=torch.float32, device=device),
            "price": torch.tensor(np.stack(batch_p), dtype=torch.float32, device=device),
            "time_delta": torch.tensor(np.stack(batch_td), dtype=torch.float32, device=device),
            "hours_to_res": torch.tensor(np.stack(batch_hr), dtype=torch.float32, device=device),
            "n_buckets": torch.tensor(np.stack(batch_nb), dtype=torch.float32, device=device),
        }
        targets = {
            "wallet": torch.tensor(np.stack(tgt_w), dtype=torch.long, device=device),
            "bucket": torch.tensor(np.stack(tgt_bp), dtype=torch.float32, device=device),
            "time": torch.tensor(np.stack(tgt_td), dtype=torch.float32, device=device),
        }
        return inputs, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: CrossMarketTPP,
    dataset: EventWindowDataset,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    patience: int = 25,
    val_fraction: float = 0.1,
) -> list[dict[str, float]]:
    """Train the cross-market TPP model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 20
    )

    n_windows = len(dataset)
    n_val = max(1, int(n_windows * val_fraction))
    n_train = n_windows - n_val
    all_indices = list(range(n_windows))

    # Last windows are validation (chronologically later)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    print(f"  Windows: {n_train} train, {n_val} val (of {n_windows} total)")

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        np.random.shuffle(train_indices)
        train_loss = 0.0
        train_n = 0

        for batch_start in range(0, n_train, batch_size):
            batch_idx = train_indices[batch_start : batch_start + batch_size]
            inputs, targets = dataset.get_batch(batch_idx, device)

            optimizer.zero_grad()
            wallet_logits, bucket_pred, time_mu, time_log_sigma = model(**inputs)
            loss, _ = compute_loss(
                wallet_logits, bucket_pred, time_mu, time_log_sigma,
                targets["wallet"], targets["bucket"], targets["time"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += float(loss) * len(batch_idx)
            train_n += len(batch_idx)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_metrics_agg: dict[str, float] = {}

        with torch.no_grad():
            for batch_start in range(0, n_val, batch_size):
                batch_idx = val_indices[batch_start : batch_start + batch_size]
                inputs, targets = dataset.get_batch(batch_idx, device)

                wallet_logits, bucket_pred, time_mu, time_log_sigma = model(**inputs)
                loss, metrics = compute_loss(
                    wallet_logits, bucket_pred, time_mu, time_log_sigma,
                    targets["wallet"], targets["bucket"], targets["time"],
                )
                val_loss += float(loss) * len(batch_idx)
                val_n += len(batch_idx)
                for k, v in metrics.items():
                    val_metrics_agg[k] = val_metrics_agg.get(k, 0) + v * len(batch_idx)

        avg_train = train_loss / max(train_n, 1)
        avg_val = val_loss / max(val_n, 1)
        val_metrics_avg = {k: v / max(val_n, 1) for k, v in val_metrics_agg.items()}

        history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            **{f"val_{k}": v for k, v in val_metrics_avg.items()},
        })

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
                f"(wallet={val_metrics_avg.get('wallet_nll', 0):.3f} "
                f"bucket={val_metrics_avg.get('bucket_mse', 0):.4f} "
                f"time={val_metrics_avg.get('time_nll', 0):.3f}) "
                f"best={best_val_loss:.4f}@{best_epoch} lr={lr_now:.2e}"
            )

        if epoch - best_epoch > patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  Restored best model from epoch {best_epoch} (val={best_val_loss:.4f})")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_held_out(
    model: CrossMarketTPP,
    data: dict,
    device: torch.device,
    context_len: int = 512,
    batch_size: int = 32,
) -> dict:
    """Evaluate on held-out test set. Returns per-event and per-market metrics."""
    model.eval()
    split = data["split_idx"]
    N = data["N"]

    # Create test windows
    test_dataset = EventWindowDataset(data, split, N, context_len)

    total_wallet_nll = 0.0
    total_bucket_mse = 0.0
    total_time_nll = 0.0
    total_events = 0

    # Per-market tracking
    per_market: dict[str, dict[str, float]] = {}

    with torch.no_grad():
        for batch_start in range(0, len(test_dataset), batch_size):
            batch_idx = list(range(
                batch_start, min(batch_start + batch_size, len(test_dataset))
            ))
            inputs, targets = test_dataset.get_batch(batch_idx, device)

            wallet_logits, bucket_pred, time_mu, time_log_sigma = model(**inputs)
            _, metrics = compute_loss(
                wallet_logits, bucket_pred, time_mu, time_log_sigma,
                targets["wallet"], targets["bucket"], targets["time"],
            )

            n_events = len(batch_idx) * context_len
            total_wallet_nll += metrics["wallet_nll"] * n_events
            total_bucket_mse += metrics["bucket_mse"] * n_events
            total_time_nll += metrics["time_nll"] * n_events
            total_events += n_events

    avg_wallet = total_wallet_nll / max(total_events, 1)
    avg_bucket = total_bucket_mse / max(total_events, 1)
    avg_time = total_time_nll / max(total_events, 1)
    avg_total = avg_wallet + avg_bucket + avg_time

    # Also evaluate per-market
    for event_id in sorted(data["event_meta"].keys()):
        # Find test events for this market
        mask = (data["event_ids"][split:N] == event_id)
        n_market = int(mask.sum())
        if n_market < context_len:
            continue

        # Find contiguous chunks of this market's events in the test set
        market_indices = np.where(mask)[0] + split
        if len(market_indices) < context_len + 1:
            continue

        # Use first chunk
        start = int(market_indices[0])
        end = min(int(market_indices[-1]) + 1, start + context_len + 1)
        if end - start < context_len + 1:
            continue

        L = context_len
        w = data["wallet_indices"][start : start + L][np.newaxis]
        c = data["city_indices"][start : start + L][np.newaxis]
        s = data["side_indices"][start : start + L][np.newaxis]
        bp = data["bucket_positions"][start : start + L][np.newaxis]
        p = data["prices"][start : start + L][np.newaxis]
        td = data["time_deltas"][start : start + L][np.newaxis]
        hr = data["hours_to_resolution"][start : start + L][np.newaxis]
        nb = data["n_buckets"][start : start + L][np.newaxis]

        inp = {
            "wallet_idx": torch.tensor(w, dtype=torch.long, device=device),
            "city_idx": torch.tensor(c, dtype=torch.long, device=device),
            "side_idx": torch.tensor(s, dtype=torch.long, device=device),
            "bucket_pos": torch.tensor(bp, dtype=torch.float32, device=device),
            "price": torch.tensor(p, dtype=torch.float32, device=device),
            "time_delta": torch.tensor(td, dtype=torch.float32, device=device),
            "hours_to_res": torch.tensor(hr, dtype=torch.float32, device=device),
            "n_buckets": torch.tensor(nb, dtype=torch.float32, device=device),
        }

        tgt_w = data["wallet_indices"][start + 1 : start + L + 1][np.newaxis]
        tgt_bp = data["bucket_positions"][start + 1 : start + L + 1][np.newaxis]
        tgt_td = data["time_deltas"][start + 1 : start + L + 1][np.newaxis]

        wallet_logits, bucket_pred, time_mu, time_log_sigma = model(**inp)
        _, m = compute_loss(
            wallet_logits, bucket_pred, time_mu, time_log_sigma,
            torch.tensor(tgt_w, dtype=torch.long, device=device),
            torch.tensor(tgt_bp, dtype=torch.float32, device=device),
            torch.tensor(tgt_td, dtype=torch.float32, device=device),
        )

        city = data["event_meta"].get(event_id, {}).get("city", "unknown")
        title = data["event_meta"].get(event_id, {}).get("event_title", event_id)
        per_market[event_id] = {
            "city": city,
            "title": title,
            "n_test_events": n_market,
            "total_nll": m["total_nll"],
            "wallet_nll": m["wallet_nll"],
            "bucket_mse": m["bucket_mse"],
            "time_nll": m["time_nll"],
        }

    return {
        "global": {
            "total_nll": avg_total,
            "wallet_nll": avg_wallet,
            "bucket_mse": avg_bucket,
            "time_nll": avg_time,
            "n_test_events": total_events,
            # Negative NLL = LL for comparison with classical
            "held_out_avg_ll": -avg_total,
        },
        "per_market": per_market,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train", alpha=0.8)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total NLL")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h.get("val_val_wallet_nll", h.get("val_wallet_nll", 0)) for h in history], label="Wallet CE", color="tab:blue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Wallet Prediction Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, [h.get("val_val_time_nll", h.get("val_time_nll", 0)) for h in history], label="Time NLL", color="tab:red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("NLL")
    axes[2].set_title("Time Prediction Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'training_curves.png'}")


def plot_wallet_embeddings(
    model: CrossMarketTPP,
    data: dict,
    output_dir: Path,
    n_wallets_plot: int = 200,
) -> np.ndarray:
    """t-SNE visualization of wallet embeddings. Returns embedding matrix."""
    model.eval()
    with torch.no_grad():
        # Get all wallet embeddings (excluding UNK)
        all_embeds = model.wallet_embed.weight.cpu().numpy()

    n_real = len(data["wallet_list"])
    embeds = all_embeds[:n_real]  # exclude UNK

    # Compute wallet statistics for coloring
    wallet_counts = np.zeros(n_real)
    wallet_n_markets = np.zeros(n_real)
    for i, wallet in enumerate(data["wallet_list"]):
        mask = data["wallet_indices"] == i
        wallet_counts[i] = mask.sum()
        wallet_n_markets[i] = len(set(data["event_ids"][mask].tolist()))

    # Take top wallets by count for visualization clarity
    top_idx = np.argsort(wallet_counts)[::-1][:n_wallets_plot]
    top_embeds = embeds[top_idx]
    top_counts = wallet_counts[top_idx]
    top_n_markets = wallet_n_markets[top_idx]

    # t-SNE
    perplexity = min(30, len(top_embeds) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(top_embeds)

    # Plot 1: colored by activity level
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc1 = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=np.log1p(top_counts), cmap="viridis",
        s=20, alpha=0.7,
    )
    plt.colorbar(sc1, ax=axes[0], label="log(1 + event count)")
    axes[0].set_title("Wallet Embeddings by Activity Level")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Plot 2: colored by number of markets
    sc2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=top_n_markets, cmap="plasma",
        s=20, alpha=0.7,
    )
    plt.colorbar(sc2, ax=axes[1], label="Number of markets")
    axes[1].set_title("Wallet Embeddings by Market Diversity")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_dir / "wallet_embeddings_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'wallet_embeddings_tsne.png'}")

    return all_embeds


def plot_attention_analysis(
    model: CrossMarketTPP,
    data: dict,
    device: torch.device,
    output_dir: Path,
    context_len: int = 128,
) -> None:
    """Analyze and plot attention patterns for cross-market influence."""
    model.eval()
    split = data["split_idx"]

    # Use a chunk from the test set
    start = split
    L = min(context_len, data["N"] - split - 1)
    if L < 32:
        print("  Skipping attention analysis: not enough test data")
        return

    inp = {
        "wallet_idx": torch.tensor(data["wallet_indices"][start:start+L][np.newaxis], dtype=torch.long, device=device),
        "city_idx": torch.tensor(data["city_indices"][start:start+L][np.newaxis], dtype=torch.long, device=device),
        "side_idx": torch.tensor(data["side_indices"][start:start+L][np.newaxis], dtype=torch.long, device=device),
        "bucket_pos": torch.tensor(data["bucket_positions"][start:start+L][np.newaxis], dtype=torch.float32, device=device),
        "price": torch.tensor(data["prices"][start:start+L][np.newaxis], dtype=torch.float32, device=device),
        "time_delta": torch.tensor(data["time_deltas"][start:start+L][np.newaxis], dtype=torch.float32, device=device),
        "hours_to_res": torch.tensor(data["hours_to_resolution"][start:start+L][np.newaxis], dtype=torch.float32, device=device),
        "n_buckets": torch.tensor(data["n_buckets"][start:start+L][np.newaxis], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        attn_weights = model.get_attention_weights(**inp)

    # Average attention across heads and layers
    avg_attn = torch.stack(attn_weights).mean(dim=(0, 1, 2)).cpu().numpy()  # (L, L)

    # Get market labels for the events
    event_ids_chunk = data["event_ids"][start:start+L]
    cities_chunk = [data["event_meta"].get(eid, {}).get("city", "?") for eid in event_ids_chunk]

    # Plot 1: Attention heatmap (last 64 events for readability)
    plot_L = min(64, L)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(avg_attn[-plot_L:, -plot_L:], cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_title("Average Attention Weights (last 64 events)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'attention_heatmap.png'}")

    # Plot 2: Cross-market attention analysis
    # For each event, how much attention goes to same-market vs different-market events
    unique_eids = list(set(event_ids_chunk.tolist()))
    if len(unique_eids) > 1:
        same_market_attn = []
        cross_market_attn = []

        for i in range(L):
            eid_i = event_ids_chunk[i]
            same_mask = event_ids_chunk[:i+1] == eid_i
            if same_mask.sum() > 0 and (~same_mask[:i+1]).sum() > 0:
                row = avg_attn[i, :i+1]
                row_sum = row.sum()
                if row_sum > 0:
                    same_frac = row[same_mask[:i+1]].sum() / row_sum
                    same_market_attn.append(same_frac)
                    cross_market_attn.append(1 - same_frac)

        if same_market_attn:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(same_market_attn, label="Same market", alpha=0.7)
            ax.plot(cross_market_attn, label="Cross market", alpha=0.7)
            ax.set_xlabel("Event index")
            ax.set_ylabel("Attention fraction")
            ax.set_title("Same-market vs Cross-market Attention")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "cross_market_attention.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {output_dir / 'cross_market_attention.png'}")


def plot_market_comparison(
    eval_results: dict,
    classical_results: dict[str, float],
    output_dir: Path,
) -> None:
    """Plot neural vs classical LL comparison per market."""
    markets = []
    neural_lls = []
    classical_lls = []
    cities = []

    for eid, mdata in eval_results["per_market"].items():
        if eid in classical_results:
            markets.append(eid)
            neural_lls.append(-mdata["total_nll"])  # negative NLL = LL
            classical_lls.append(classical_results[eid])
            cities.append(mdata["city"])

    if not markets:
        print("  No overlapping markets for comparison plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(markets))
    width = 0.35

    # Color by city
    unique_cities = sorted(set(cities))
    city_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_cities)))
    city_cmap = {c: city_colors[i] for i, c in enumerate(unique_cities)}
    colors = [city_cmap[c] for c in cities]

    ax.bar(x - width/2, classical_lls, width, label="Classical Hawkes", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, neural_lls, width, label="Neural TPP", color="coral", alpha=0.8)

    ax.set_xlabel("Market")
    ax.set_ylabel("Held-out avg LL")
    ax.set_title("Per-Market: Classical Hawkes vs Neural TPP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eid}\n({c})" for eid, c in zip(markets, cities)],
                       rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "market_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'market_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-market Transformer TPP")
    parser.add_argument("--events-dir", type=Path, default=Path("data/events"))
    parser.add_argument("--out", type=Path, default=Path("data/reports/generalization/cross_market"))
    parser.add_argument("--top-wallets", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--wallet-dim", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70)
    print("Cross-Market Transformer TPP with Wallet Embeddings")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("\nWARNING: CUDA not available, using CPU")

    # 1. Load data
    print("\n[1/6] Loading and preprocessing cross-market data...")
    data = load_cross_market_data(args.events_dir, top_k_wallets=args.top_wallets)

    # 2. Build model
    print(f"\n[2/6] Building model (d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads})...")
    model = CrossMarketTPP(
        n_wallets=data["n_wallets"],
        n_cities=data["n_cities"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        wallet_dim=args.wallet_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Wallet vocab: {data['n_wallets']} ({len(data['wallet_list'])} + UNK)")
    print(f"  City vocab: {data['n_cities']}")

    # 3. Train
    print(f"\n[3/6] Training (epochs={args.epochs}, lr={args.lr}, ctx={args.context_len})...")
    train_dataset = EventWindowDataset(
        data, 0, data["split_idx"], context_len=args.context_len
    )

    t0 = time.time()
    history = train_model(
        model, train_dataset, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    vram_used = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  Peak VRAM: {vram_used:.1f} MB")

    # 4. Evaluate
    print("\n[4/6] Evaluating held-out performance...")
    eval_results = evaluate_held_out(
        model, data, device,
        context_len=args.context_len, batch_size=args.batch_size,
    )

    g = eval_results["global"]
    print(f"  Global held-out NLL: {g['total_nll']:.4f}")
    print(f"    Wallet NLL:  {g['wallet_nll']:.4f}")
    print(f"    Bucket MSE:  {g['bucket_mse']:.4f}")
    print(f"    Time NLL:    {g['time_nll']:.4f}")
    print(f"  Global held-out LL:  {g['held_out_avg_ll']:.4f}")

    # Load classical results for comparison
    classical_results: dict[str, float] = {}
    gen_dir = args.out.parent
    for market_dir in gen_dir.iterdir():
        results_path = market_dir / "results.json"
        if results_path.exists() and market_dir.name != "cross_market":
            try:
                with open(results_path) as f:
                    cr = json.load(f)
                if cr.get("model") == "multivariate_exponential_hawkes":
                    classical_results[cr["event_id"]] = cr["metrics"]["held_out_avg_log_likelihood"]
            except Exception:
                pass

    if classical_results:
        classical_mean = np.mean(list(classical_results.values()))
        print(f"\n  Classical baseline (mean over {len(classical_results)} markets): {classical_mean:.4f}")

        # Per-market comparison
        print("\n  Per-market comparison:")
        neural_beats = 0
        n_compared = 0
        for eid, mdata in eval_results["per_market"].items():
            if eid in classical_results:
                neural_ll = -mdata["total_nll"]
                classical_ll = classical_results[eid]
                beats = neural_ll > classical_ll
                if beats:
                    neural_beats += 1
                n_compared += 1
                marker = ">>>" if beats else "   "
                print(f"    {marker} {eid} ({mdata['city']:>8s}): neural={neural_ll:.3f} classical={classical_ll:.3f}")
        if n_compared > 0:
            print(f"\n  Neural beats classical: {neural_beats}/{n_compared} markets")

    # 5. Visualizations
    print("\n[5/6] Generating visualizations...")
    out_dir = args.out
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(history, plots_dir)

    embeds = plot_wallet_embeddings(model, data, plots_dir)

    plot_attention_analysis(model, data, device, plots_dir)

    if classical_results:
        plot_market_comparison(eval_results, classical_results, plots_dir)

    # 6. Save results
    print("\n[6/6] Saving results...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"  Saved: {out_dir / 'model.pt'}")

    # Wallet embeddings
    np.save(out_dir / "embeddings.npy", embeds)
    print(f"  Saved: {out_dir / 'embeddings.npy'}")

    # Results JSON
    results = {
        "model": "cross_market_transformer_tpp",
        "architecture": {
            "type": "Transformer TPP with entity embeddings",
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "wallet_dim": args.wallet_dim,
            "context_len": args.context_len,
        },
        "n_parameters": n_params,
        "data": {
            "n_markets": len(data["event_meta"]),
            "n_events_total": data["N"],
            "n_train": data["split_idx"],
            "n_test": data["N"] - data["split_idx"],
            "n_wallets_vocab": data["n_wallets"],
            "n_cities": data["n_cities"],
            "cities": data["city_list"],
        },
        "training": {
            "epochs_run": len(history),
            "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"],
            "best_val_loss": min(h["val_loss"] for h in history),
            "final_train_loss": history[-1]["train_loss"],
            "lr": args.lr,
            "batch_size": args.batch_size,
            "training_time_seconds": train_time,
        },
        "metrics": {
            "global_held_out_nll": g["total_nll"],
            "global_held_out_ll": g["held_out_avg_ll"],
            "wallet_nll": g["wallet_nll"],
            "bucket_mse": g["bucket_mse"],
            "time_nll": g["time_nll"],
            "n_test_events": g["n_test_events"],
        },
        "comparison": {
            "classical_mean_ll": float(np.mean(list(classical_results.values()))) if classical_results else None,
            "classical_n_markets": len(classical_results),
            "per_market": {
                eid: {
                    "neural_ll": -mdata["total_nll"],
                    "classical_ll": classical_results.get(eid),
                    "city": mdata["city"],
                }
                for eid, mdata in eval_results["per_market"].items()
            },
        },
        "gpu": {
            "name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu",
            "peak_vram_mb": vram_used,
        },
        "wallet_vocabulary": {
            "size": len(data["wallet_list"]),
            "top_10": [
                {"idx": i, "address": data["wallet_list"][i][:16] + "..."}
                for i in range(min(10, len(data["wallet_list"])))
            ],
        },
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / 'results.json'}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Model:                 Cross-market Transformer TPP")
    print(f"  Parameters:            {n_params:,}")
    print(f"  Markets:               {len(data['event_meta'])}")
    print(f"  Events:                {data['N']:,} ({data['split_idx']:,} train / {data['N']-data['split_idx']:,} test)")
    print(f"  Wallet vocab:          {data['n_wallets']}")
    print(f"  Training epochs:       {len(history)}")
    print(f"  Held-out LL:           {g['held_out_avg_ll']:.4f}")
    if classical_results:
        print(f"  Classical baseline LL: {float(np.mean(list(classical_results.values()))):.4f}")
    if torch.cuda.is_available():
        print(f"  GPU:                   {torch.cuda.get_device_name(args.gpu)}")
        print(f"  Peak VRAM:             {vram_used:.1f} MB")
    print(f"  Output:                {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
