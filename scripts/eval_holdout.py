"""Evaluate the trained cross-market transformer on held-out March 25-31 events.

Loads the pre-trained model and vocabulary, evaluates on all events that
ended after the training cutoff (March 25, 2026). Fits per-market classical
Hawkes baselines in parallel for comparison.

Outputs per-city, per-date held-out LL and comparison to classical baseline.

Usage:
    uv run python scripts/eval_holdout.py [--workers 4]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Reuse model + helpers from training script
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
        self.wallet_embed = nn.Embedding(n_wallets, wallet_dim)
        self.city_embed = nn.Embedding(n_cities, city_dim)
        self.side_embed = nn.Embedding(2, side_dim)
        self.bucket_proj = nn.Linear(1, 16)
        self.price_proj = nn.Linear(1, 8)
        self.context_proj = nn.Linear(2, 8)
        self.time_dim = time_dim
        feat_dim = wallet_dim + city_dim + side_dim + 16 + 8 + 8 + time_dim
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.wallet_head = nn.Linear(d_model, n_wallets)
        self.bucket_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))
        self.time_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, wallet_idx, city_idx, side_idx, bucket_pos, price,
                time_delta, hours_to_res, n_buckets):
        B, L = wallet_idx.shape
        w_emb = self.wallet_embed(wallet_idx)
        c_emb = self.city_embed(city_idx)
        s_emb = self.side_embed(side_idx)
        b_enc = self.bucket_proj(bucket_pos.unsqueeze(-1))
        p_enc = self.price_proj(price.unsqueeze(-1))
        ctx = torch.stack([hours_to_res / 200.0, n_buckets / 11.0], dim=-1)
        ctx_enc = self.context_proj(ctx)
        t_enc = sinusoidal_time_encoding(time_delta, self.time_dim)
        features = torch.cat([w_emb, c_emb, s_emb, b_enc, p_enc, ctx_enc, t_enc], dim=-1)
        x = self.input_proj(features)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        wallet_logits = self.wallet_head(h)
        bucket_pred = self.bucket_head(h).squeeze(-1)
        time_params = self.time_head(h)
        return wallet_logits, bucket_pred, time_params[..., 0], time_params[..., 1]


def compute_loss(wallet_logits, bucket_pred, time_mu, time_log_sigma,
                 target_wallet, target_bucket, target_time, mask=None):
    B, L = target_wallet.shape
    wallet_loss = F.cross_entropy(
        wallet_logits.reshape(-1, wallet_logits.size(-1)),
        target_wallet.reshape(-1), reduction="none",
    ).reshape(B, L)
    bucket_loss = F.mse_loss(torch.sigmoid(bucket_pred), target_bucket, reduction="none")
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
    return wallet_mean + bucket_mean + time_mean, {
        "wallet_nll": float(wallet_mean),
        "bucket_mse": float(bucket_mean),
        "time_nll": float(time_mean),
        "total_nll": float(wallet_mean + bucket_mean + time_mean),
    }


# ---------------------------------------------------------------------------
# Bucket position computation (from training script)
# ---------------------------------------------------------------------------


def parse_suit_sort_key(suit: str) -> float:
    s = suit.strip()
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return 0.0
    if s.startswith("<=") or s.startswith("<"):
        return float(nums[0]) - 0.5
    if s.startswith(">=") or s.startswith(">"):
        return float(nums[0]) + 0.5
    return float(nums[0])


def compute_bucket_positions(df: pl.DataFrame, meta_dir: Path) -> pl.DataFrame:
    positions = []
    for event_id in df["event_id"].unique().sort().to_list():
        meta_path = meta_dir / event_id / "_meta.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            suits = meta.get("bucket_labels", [])
        else:
            suits = df.filter(pl.col("event_id") == event_id)["suit"].unique().sort().to_list()
        sorted_suits = sorted(suits, key=parse_suit_sort_key)
        n = len(sorted_suits)
        suit_to_pos = {s: i / max(n - 1, 1) for i, s in enumerate(sorted_suits)}
        positions.append((event_id, suit_to_pos))

    pos_map: dict[tuple[str, str], float] = {}
    for event_id, suit_map in positions:
        for suit, pos in suit_map.items():
            pos_map[(event_id, suit)] = pos

    bucket_pos = [
        pos_map.get((eid, suit), 0.5)
        for eid, suit in zip(df["event_id"].to_list(), df["suit"].to_list())
    ]
    return df.with_columns(pl.Series("bucket_position", bucket_pos))


# ---------------------------------------------------------------------------
# Classical Hawkes baseline (simplified for batch evaluation)
# ---------------------------------------------------------------------------


def fit_classical_hawkes_single(event_id: str, events_dir: Path,
                                max_dims: int = 30, n_top_wallets: int = 30,
                                min_events_per_pair: int = 10) -> dict:
    """Fit a per-market classical Hawkes and return held-out LL."""
    parquet_path = events_dir / event_id / "events.parquet"
    if not parquet_path.exists():
        return {"event_id": event_id, "error": "no parquet", "held_out_ll": None}

    try:
        df = pl.read_parquet(parquet_path)
        if df.height < 200:
            return {"event_id": event_id, "error": "too few trades", "held_out_ll": None,
                    "n_trades": df.height}

        # Top wallets
        top_wallets = (
            df.group_by("actor").agg(pl.len().alias("n"))
            .sort("n", descending=True).head(n_top_wallets)["actor"].to_list()
        )
        df_filt = df.filter(pl.col("actor").is_in(top_wallets)).sort("seq")

        # (wallet, suit) pairs
        pairs = (
            df_filt.group_by("actor", "suit").agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_events_per_pair)
            .sort("n", descending=True).head(max_dims)
        )
        if pairs.height < 2:
            return {"event_id": event_id, "error": "too few dims", "held_out_ll": None}

        dim_map: dict[tuple[str, str], int] = {}
        for row in pairs.iter_rows(named=True):
            dim_map[(row["actor"], row["suit"])] = len(dim_map)
        D = len(dim_map)

        times_list, dims_list = [], []
        for row in df_filt.sort("seq").iter_rows(named=True):
            key = (row["actor"], row["suit"])
            if key in dim_map:
                times_list.append(float(row["seq"]))
                dims_list.append(dim_map[key])

        times = np.array(times_list, dtype=np.float64)
        dims = np.array(dims_list, dtype=np.int32)
        N = len(times)
        split = int(0.8 * N)
        if split < 50 or N - split < 20:
            return {"event_id": event_id, "error": "split too small", "held_out_ll": None}

        T_train = times[split - 1] + 1.0
        T_all = times[-1] + 1.0

        times_train, dims_train = times[:split], dims[:split]
        times_test, dims_test = times[split:], dims[split:]

        # Fit with grid search over beta
        best_nll = np.inf
        best_params = None
        best_beta = None

        for beta_val in [0.1, 0.5]:
            log_beta = np.log(beta_val)
            mu_init = np.array([max(np.sum(dims_train == d) / T_train, 1e-6) for d in range(D)])
            alpha_init = np.full(D * D, 0.001)
            x0 = np.concatenate([mu_init, alpha_init, [log_beta]])
            bounds = [(1e-8, None)] * D + [(0.0, 1.0)] * (D * D) + [(log_beta, log_beta)]

            result = minimize(
                _neg_log_likelihood, x0,
                args=(times_train, dims_train, D, T_train, 0.05),
                method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
                best_beta = beta_val

        mu = best_params[:D]
        alpha = best_params[D:D + D * D].reshape(D, D)
        beta = best_beta

        # Evaluate on held-out set
        ll = _eval_log_likelihood(
            mu, alpha, beta, times_train, dims_train,
            times_test, dims_test, T_all, D,
        )

        branching = float(np.max(np.abs(np.linalg.eigvals(alpha))))

        return {
            "event_id": event_id,
            "held_out_ll": float(ll),
            "branching_ratio": branching,
            "dimensions": D,
            "n_train": split,
            "n_test": N - split,
            "beta": beta,
        }

    except Exception as e:
        return {"event_id": event_id, "error": str(e)[:200], "held_out_ll": None}


def _neg_log_likelihood(params, times, dims, D, T, l1_penalty=0.0):
    mu = params[:D]
    alpha = params[D:D + D * D].reshape(D, D)
    log_beta = params[D + D * D]
    beta = np.exp(log_beta)
    N = len(times)
    A = np.zeros(D, dtype=np.float64)
    grad_mu = np.zeros(D, dtype=np.float64)
    grad_alpha = np.zeros((D, D), dtype=np.float64)
    ll = 0.0
    prev_t = times[0]
    for n in range(N):
        t_n = times[n]
        d_n = dims[n]
        if n > 0:
            dt = t_n - prev_t
            decay = np.exp(-beta * dt)
            indicator = np.zeros(D)
            indicator[dims[n - 1]] = 1.0
            A = decay * (A + indicator)
        lam_n = mu[d_n] + beta * np.dot(alpha[d_n], A)
        if lam_n <= 1e-15:
            lam_n = 1e-15
        ll += np.log(lam_n)
        inv_lam = 1.0 / lam_n
        grad_mu[d_n] += inv_lam
        grad_alpha[d_n] += beta * A * inv_lam
        prev_t = t_n
    compensator_alpha = np.zeros((D, D), dtype=np.float64)
    for n in range(N):
        j = dims[n]
        c = 1.0 - np.exp(-beta * (T - times[n]))
        compensator_alpha[:, j] += c
    ll -= np.sum(mu * T)
    ll -= np.sum(alpha * compensator_alpha)
    grad_mu -= T
    grad_alpha -= compensator_alpha
    if l1_penalty > 0:
        ll -= l1_penalty * np.sum(alpha)
        grad_alpha -= l1_penalty
    grad = np.zeros_like(params)
    grad[:D] = -grad_mu
    grad[D:D + D * D] = -grad_alpha.ravel()
    grad[D + D * D] = 0.0
    return -ll, grad


def _eval_log_likelihood(mu, alpha, beta, times_train, dims_train,
                         times_test, dims_test, T_test_end, D):
    A = np.zeros(D, dtype=np.float64)
    if len(times_train) > 0:
        prev_t = times_train[0]
        for n in range(len(times_train)):
            t_n = times_train[n]
            if n > 0:
                dt = t_n - prev_t
                decay = np.exp(-beta * dt)
                indicator = np.zeros(D)
                indicator[dims_train[n - 1]] = 1.0
                A = decay * (A + indicator)
            prev_t = t_n
    ll = 0.0
    prev_t = times_train[-1] if len(times_train) > 0 else times_test[0]
    N_test = len(times_test)
    for n in range(N_test):
        t_n = times_test[n]
        d_n = dims_test[n]
        dt = t_n - prev_t
        decay = np.exp(-beta * dt)
        if n == 0:
            indicator = np.zeros(D)
            indicator[dims_train[-1]] = 1.0
        else:
            indicator = np.zeros(D)
            indicator[dims_test[n - 1]] = 1.0
        A = decay * (A + indicator)
        lam_n = mu[d_n] + beta * np.dot(alpha[d_n], A)
        if lam_n <= 1e-15:
            lam_n = 1e-15
        ll += np.log(lam_n)
        prev_t = t_n
    T_test_start = times_test[0]
    T_duration = T_test_end - T_test_start
    for i in range(D):
        ll -= mu[i] * T_duration
    for n in range(N_test):
        j = dims_test[n]
        c = 1.0 - np.exp(-beta * (T_test_end - times_test[n]))
        ll -= np.sum(alpha[:, j]) * c
    for n in range(len(times_train)):
        j = dims_train[n]
        c_end = 1.0 - np.exp(-beta * (T_test_end - times_train[n]))
        c_start = 1.0 - np.exp(-beta * (T_test_start - times_train[n]))
        ll -= np.sum(alpha[:, j]) * (c_end - c_start)
    return ll / N_test


# ---------------------------------------------------------------------------
# Neural model evaluation per event
# ---------------------------------------------------------------------------


def evaluate_neural_per_event(
    model: CrossMarketTPP,
    event_id: str,
    events_dir: Path,
    wallet_to_idx: dict[str, int],
    city_to_idx: dict[str, int],
    unk_wallet_idx: int,
    device: torch.device,
    context_len: int = 512,
) -> dict | None:
    """Evaluate the neural model on a single event's trade sequence."""
    parquet_path = events_dir / event_id / "events.parquet"
    meta_path = events_dir / event_id / "_meta.json"
    if not parquet_path.exists():
        return None

    df = pl.read_parquet(parquet_path).sort("seq")
    if df.height < context_len + 1:
        return None

    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    city = meta.get("city", "unknown")
    date = meta.get("date", "unknown")

    # Compute bucket positions
    df = compute_bucket_positions(df, events_dir)

    # Map features
    actors = df["actor"].to_list()
    event_ids = df["event_id"].to_list()
    timestamps = df["timestamp_ms"].to_numpy().astype(np.float64)
    bucket_positions = df["bucket_position"].to_numpy().astype(np.float32)
    prices = df["price"].to_numpy().astype(np.float32)
    sides = df["side"].to_list()

    wallet_indices = np.array(
        [wallet_to_idx.get(a, unk_wallet_idx) for a in actors], dtype=np.int64
    )
    # Map unseen cities to 0 (global) as fallback
    city_idx_val = city_to_idx.get(city, 0)
    city_indices = np.full(len(actors), city_idx_val, dtype=np.int64)
    side_indices = np.array([0 if s == "BUY" else 1 for s in sides], dtype=np.int64)

    # Time deltas in hours
    time_hours = (timestamps - timestamps[0]) / 3_600_000.0
    time_deltas = np.zeros_like(time_hours)
    time_deltas[1:] = np.diff(time_hours)
    time_deltas = np.clip(time_deltas, 0, None)

    # Hours to resolution
    end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
    hours_to_resolution = np.zeros(len(actors), dtype=np.float32)
    if end_s > 0:
        for i, ts in enumerate(timestamps):
            hours_to_resolution[i] = max(0, (end_s * 1000 - ts) / 3_600_000.0)

    n_buckets_val = meta.get("n_buckets", 11)
    n_buckets_arr = np.full(len(actors), float(n_buckets_val), dtype=np.float32)

    # Evaluate in non-overlapping windows (using last 20% as test, like training)
    N = len(wallet_indices)
    split = int(0.8 * N)
    if N - split < context_len + 1:
        return None

    total_nll = 0.0
    total_wallet_nll = 0.0
    total_bucket_mse = 0.0
    total_time_nll = 0.0
    n_windows = 0

    model.eval()
    with torch.no_grad():
        i = split
        while i + context_len + 1 <= N:
            L = context_len
            inp = {
                "wallet_idx": torch.tensor(wallet_indices[i:i+L][np.newaxis], dtype=torch.long, device=device),
                "city_idx": torch.tensor(city_indices[i:i+L][np.newaxis], dtype=torch.long, device=device),
                "side_idx": torch.tensor(side_indices[i:i+L][np.newaxis], dtype=torch.long, device=device),
                "bucket_pos": torch.tensor(bucket_positions[i:i+L][np.newaxis], dtype=torch.float32, device=device),
                "price": torch.tensor(prices[i:i+L][np.newaxis], dtype=torch.float32, device=device),
                "time_delta": torch.tensor(time_deltas[i:i+L][np.newaxis], dtype=torch.float32, device=device),
                "hours_to_res": torch.tensor(hours_to_resolution[i:i+L][np.newaxis], dtype=torch.float32, device=device),
                "n_buckets": torch.tensor(n_buckets_arr[i:i+L][np.newaxis], dtype=torch.float32, device=device),
            }
            tgt_w = torch.tensor(wallet_indices[i+1:i+L+1][np.newaxis], dtype=torch.long, device=device)
            tgt_bp = torch.tensor(bucket_positions[i+1:i+L+1][np.newaxis], dtype=torch.float32, device=device)
            tgt_td = torch.tensor(time_deltas[i+1:i+L+1][np.newaxis], dtype=torch.float32, device=device)

            wl, bp, tm, tls = model(**inp)
            _, metrics = compute_loss(wl, bp, tm, tls, tgt_w, tgt_bp, tgt_td)

            total_nll += metrics["total_nll"]
            total_wallet_nll += metrics["wallet_nll"]
            total_bucket_mse += metrics["bucket_mse"]
            total_time_nll += metrics["time_nll"]
            n_windows += 1
            i += context_len

    if n_windows == 0:
        return None

    avg_nll = total_nll / n_windows
    pct_known_wallets = float(np.mean(wallet_indices[split:] != unk_wallet_idx)) * 100

    return {
        "event_id": event_id,
        "city": city,
        "date": date,
        "n_trades": N,
        "n_test": N - split,
        "n_windows": n_windows,
        "neural_ll": -avg_nll,
        "wallet_nll": total_wallet_nll / n_windows,
        "bucket_mse": total_bucket_mse / n_windows,
        "time_nll": total_time_nll / n_windows,
        "pct_known_wallets": pct_known_wallets,
        "city_known": city in city_to_idx,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_comparison_by_city(results: list[dict], output_dir: Path) -> None:
    """Bar chart: neural vs classical LL aggregated by city."""
    city_neural = defaultdict(list)
    city_classical = defaultdict(list)
    for r in results:
        city = r["city"]
        if r.get("neural_ll") is not None:
            city_neural[city].append(r["neural_ll"])
        if r.get("classical_ll") is not None:
            city_classical[city].append(r["classical_ll"])

    cities = sorted(set(city_neural.keys()) | set(city_classical.keys()),
                    key=lambda c: -len(city_neural.get(c, [])))
    if not cities:
        return

    neural_means = [np.mean(city_neural.get(c, [np.nan])) for c in cities]
    classical_means = [np.mean(city_classical.get(c, [np.nan])) for c in cities]
    counts = [len(city_neural.get(c, [])) for c in cities]

    fig, ax = plt.subplots(figsize=(max(14, len(cities) * 0.5), 7))
    x = np.arange(len(cities))
    w = 0.35
    ax.bar(x - w/2, classical_means, w, label="Classical Hawkes", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, neural_means, w, label="Neural TPP", color="coral", alpha=0.8)
    ax.set_xlabel("City")
    ax.set_ylabel("Held-out avg LL")
    ax.set_title("Per-City: Classical Hawkes vs Cross-Market Neural TPP (March 25-31)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cities, counts)],
                       rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "holdout_by_city.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'holdout_by_city.png'}")


def plot_comparison_by_date(results: list[dict], output_dir: Path) -> None:
    """Bar chart: neural vs classical LL aggregated by date."""
    date_neural = defaultdict(list)
    date_classical = defaultdict(list)
    for r in results:
        date = r["date"]
        if r.get("neural_ll") is not None:
            date_neural[date].append(r["neural_ll"])
        if r.get("classical_ll") is not None:
            date_classical[date].append(r["classical_ll"])

    dates = sorted(set(date_neural.keys()) | set(date_classical.keys()))
    if not dates:
        return

    neural_means = [np.mean(date_neural.get(d, [np.nan])) for d in dates]
    classical_means = [np.mean(date_classical.get(d, [np.nan])) for d in dates]
    counts = [len(date_neural.get(d, [])) for d in dates]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dates))
    w = 0.35
    ax.bar(x - w/2, classical_means, w, label="Classical Hawkes", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, neural_means, w, label="Neural TPP", color="coral", alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Held-out avg LL")
    ax.set_title("Per-Date: Classical Hawkes vs Cross-Market Neural TPP (March 25-31)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(dates, counts)], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "holdout_by_date.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'holdout_by_date.png'}")


def plot_scatter_comparison(results: list[dict], output_dir: Path) -> None:
    """Scatter plot: neural LL vs classical LL per event."""
    paired = [(r["neural_ll"], r["classical_ll"], r["city"], r.get("city_known", False))
              for r in results
              if r.get("neural_ll") is not None and r.get("classical_ll") is not None]
    if not paired:
        return

    neural, classical, cities, known = zip(*paired)
    neural, classical = np.array(neural), np.array(classical)

    unique_cities = sorted(set(cities))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_cities), 1)))
    city_color = {c: cmap[i] for i, c in enumerate(unique_cities)}

    fig, ax = plt.subplots(figsize=(10, 10))
    for c in unique_cities:
        mask = np.array([ci == c for ci in cities])
        ax.scatter(classical[mask], neural[mask], c=[city_color[c]], label=c,
                   s=30, alpha=0.7, edgecolors="white", linewidths=0.3)

    mn = min(classical.min(), neural.min())
    mx = max(classical.max(), neural.max())
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="y=x (tie)")
    ax.set_xlabel("Classical Hawkes held-out LL")
    ax.set_ylabel("Neural TPP held-out LL")
    ax.set_title(f"Per-Event Comparison (n={len(paired)} events)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "holdout_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'holdout_scatter.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out March 25-31 events")
    parser.add_argument("--events-dir", type=Path, default=Path("data/events"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/reports/generalization/cross_market"))
    parser.add_argument("--out", type=Path, default=Path("data/reports/generalization/cross_market"))
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for classical fits")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    CUTOFF_EPOCH = datetime.datetime(2026, 3, 25, tzinfo=datetime.timezone.utc).timestamp()

    print("=" * 70)
    print("Held-Out Evaluation: Cross-Market Neural TPP vs Classical Hawkes")
    print(f"  Cutoff: events ending >= 2026-03-25 UTC")
    print("=" * 70)

    # 1. Load vocabulary
    print("\n[1/5] Loading vocabulary and model...")
    vocab_path = args.model_dir / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    wallet_list = vocab["wallet_list"]
    city_list = vocab["city_list"]
    wallet_to_idx = {w: i for i, w in enumerate(wallet_list)}
    city_to_idx = {c: i for i, c in enumerate(city_list)}
    unk_wallet_idx = len(wallet_list)
    train_eids = set(vocab.get("train_eids", []))
    print(f"  Wallet vocab: {len(wallet_list)} + UNK")
    print(f"  City vocab: {city_list}")
    print(f"  Training events: {len(train_eids)}")

    # 2. Load model
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("  Using CPU")

    model = CrossMarketTPP(
        n_wallets=len(wallet_list) + 1,
        n_cities=len(city_list),
    ).to(device)
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location=device, weights_only=False))
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # 3. Identify held-out events
    print("\n[2/5] Identifying held-out events...")
    holdout_events = []
    for p in sorted(args.events_dir.iterdir()):
        if not p.is_dir() or p.name == "__pycache__":
            continue
        meta_path = p / "_meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        end_s = meta.get("time_range", {}).get("end_epoch_s", 0)
        if end_s >= CUTOFF_EPOCH:
            holdout_events.append({
                "event_id": p.name,
                "city": meta.get("city", "unknown"),
                "date": meta.get("date", "unknown"),
                "n_trades": meta.get("n_trades", 0),
                "in_training": p.name in train_eids,
            })

    print(f"  Found {len(holdout_events)} held-out events")
    cities = set(e["city"] for e in holdout_events)
    dates = set(e["date"] for e in holdout_events)
    in_train = sum(1 for e in holdout_events if e["in_training"])
    print(f"  Cities: {len(cities)}")
    print(f"  Dates: {len(dates)}")
    print(f"  Events also in training set: {in_train} (flagged, not excluded)")

    # 4. Evaluate neural model per event
    print(f"\n[3/5] Evaluating neural model on {len(holdout_events)} events...")
    neural_results = {}
    t0 = time.time()
    for i, evt in enumerate(holdout_events):
        eid = evt["event_id"]
        result = evaluate_neural_per_event(
            model, eid, args.events_dir,
            wallet_to_idx, city_to_idx, unk_wallet_idx,
            device, context_len=args.context_len,
        )
        if result:
            neural_results[eid] = result
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(holdout_events)}] evaluated, "
                  f"{len(neural_results)} successful ({time.time()-t0:.1f}s)")

    neural_time = time.time() - t0
    print(f"  Neural eval: {len(neural_results)}/{len(holdout_events)} events in {neural_time:.1f}s")

    # 5. Fit classical Hawkes in parallel
    print(f"\n[4/5] Fitting classical Hawkes baselines ({args.workers} workers)...")
    classical_results = {}
    t0 = time.time()
    event_ids = [e["event_id"] for e in holdout_events]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fit_classical_hawkes_single, eid, args.events_dir): eid
            for eid in event_ids
        }
        done = 0
        for future in as_completed(futures):
            eid = futures[future]
            try:
                result = future.result()
                if result.get("held_out_ll") is not None:
                    classical_results[eid] = result
            except Exception as e:
                pass
            done += 1
            if done % 50 == 0:
                print(f"    [{done}/{len(event_ids)}] fitted, "
                      f"{len(classical_results)} successful ({time.time()-t0:.1f}s)")

    classical_time = time.time() - t0
    print(f"  Classical fits: {len(classical_results)}/{len(holdout_events)} events in {classical_time:.1f}s")

    # 6. Combine results and compute aggregates
    print("\n[5/5] Computing aggregates and saving results...")
    combined = []
    for evt in holdout_events:
        eid = evt["event_id"]
        entry = {
            "event_id": eid,
            "city": evt["city"],
            "date": evt["date"],
            "n_trades": evt["n_trades"],
            "in_training": evt["in_training"],
            "neural_ll": neural_results.get(eid, {}).get("neural_ll"),
            "classical_ll": classical_results.get(eid, {}).get("held_out_ll"),
            "pct_known_wallets": neural_results.get(eid, {}).get("pct_known_wallets"),
            "city_known": evt["city"] in city_to_idx,
        }
        combined.append(entry)

    # Per-city aggregates
    per_city: dict[str, dict] = {}
    for city in sorted(cities):
        city_events = [r for r in combined if r["city"] == city]
        neural_lls = [r["neural_ll"] for r in city_events if r["neural_ll"] is not None]
        classical_lls = [r["classical_ll"] for r in city_events if r["classical_ll"] is not None]
        per_city[city] = {
            "n_events": len(city_events),
            "n_neural_eval": len(neural_lls),
            "n_classical_eval": len(classical_lls),
            "neural_mean_ll": float(np.mean(neural_lls)) if neural_lls else None,
            "classical_mean_ll": float(np.mean(classical_lls)) if classical_lls else None,
            "city_in_training": city in city_to_idx,
        }

    # Per-date aggregates
    per_date: dict[str, dict] = {}
    for date in sorted(dates):
        date_events = [r for r in combined if r["date"] == date]
        neural_lls = [r["neural_ll"] for r in date_events if r["neural_ll"] is not None]
        classical_lls = [r["classical_ll"] for r in date_events if r["classical_ll"] is not None]
        per_date[date] = {
            "n_events": len(date_events),
            "n_neural_eval": len(neural_lls),
            "n_classical_eval": len(classical_lls),
            "neural_mean_ll": float(np.mean(neural_lls)) if neural_lls else None,
            "classical_mean_ll": float(np.mean(classical_lls)) if classical_lls else None,
        }

    # Overall aggregates
    all_neural = [r["neural_ll"] for r in combined if r["neural_ll"] is not None]
    all_classical = [r["classical_ll"] for r in combined if r["classical_ll"] is not None]
    paired = [(r["neural_ll"], r["classical_ll"])
              for r in combined
              if r["neural_ll"] is not None and r["classical_ll"] is not None]

    neural_beats = sum(1 for n, c in paired if n > c)

    # Separate seen vs unseen cities
    seen_neural = [r["neural_ll"] for r in combined if r["neural_ll"] is not None and r["city_known"]]
    unseen_neural = [r["neural_ll"] for r in combined if r["neural_ll"] is not None and not r["city_known"]]

    summary = {
        "eval_type": "holdout_march_25_31",
        "cutoff_date": "2026-03-25",
        "n_holdout_events": len(holdout_events),
        "n_neural_evaluated": len(all_neural),
        "n_classical_fitted": len(all_classical),
        "n_paired_comparison": len(paired),
        "overall": {
            "neural_mean_ll": float(np.mean(all_neural)) if all_neural else None,
            "neural_std_ll": float(np.std(all_neural)) if all_neural else None,
            "classical_mean_ll": float(np.mean(all_classical)) if all_classical else None,
            "classical_std_ll": float(np.std(all_classical)) if all_classical else None,
            "neural_beats_classical": neural_beats,
            "neural_beats_pct": round(100 * neural_beats / max(len(paired), 1), 1),
        },
        "generalization": {
            "n_seen_cities": len([c for c in cities if c in city_to_idx]),
            "n_unseen_cities": len([c for c in cities if c not in city_to_idx]),
            "seen_city_neural_mean_ll": float(np.mean(seen_neural)) if seen_neural else None,
            "unseen_city_neural_mean_ll": float(np.mean(unseen_neural)) if unseen_neural else None,
        },
        "timing": {
            "neural_eval_seconds": round(neural_time, 1),
            "classical_fit_seconds": round(classical_time, 1),
        },
        "per_city": per_city,
        "per_date": per_date,
        "per_event": combined,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("HELD-OUT EVALUATION RESULTS (March 25-31, 2026)")
    print("=" * 70)
    o = summary["overall"]
    print(f"  Events evaluated: {len(all_neural)} neural, {len(all_classical)} classical")
    print(f"  Neural mean LL:    {o['neural_mean_ll']:.4f} (std {o['neural_std_ll']:.4f})" if o['neural_mean_ll'] else "  Neural: N/A")
    print(f"  Classical mean LL: {o['classical_mean_ll']:.4f} (std {o['classical_std_ll']:.4f})" if o['classical_mean_ll'] else "  Classical: N/A")
    print(f"  Neural beats classical: {o['neural_beats_classical']}/{len(paired)} ({o['neural_beats_pct']}%)")

    g = summary["generalization"]
    print(f"\n  City generalization:")
    print(f"    Seen cities ({g['n_seen_cities']}):   neural mean LL = {g['seen_city_neural_mean_ll']:.4f}" if g['seen_city_neural_mean_ll'] else "")
    print(f"    Unseen cities ({g['n_unseen_cities']}): neural mean LL = {g['unseen_city_neural_mean_ll']:.4f}" if g['unseen_city_neural_mean_ll'] else "")

    print(f"\n  Per-city (top 10 by event count):")
    for city in sorted(per_city.keys(), key=lambda c: -per_city[c]["n_events"])[:10]:
        c = per_city[city]
        known = "SEEN" if c["city_in_training"] else "NEW "
        nll = f"neural={c['neural_mean_ll']:.3f}" if c['neural_mean_ll'] is not None else "neural=N/A"
        cll = f"classical={c['classical_mean_ll']:.3f}" if c['classical_mean_ll'] is not None else "classical=N/A"
        print(f"    [{known}] {city:15s} ({c['n_events']:2d} events): {nll}  {cll}")

    print(f"\n  Per-date:")
    for date in sorted(per_date.keys()):
        d = per_date[date]
        nll = f"neural={d['neural_mean_ll']:.3f}" if d['neural_mean_ll'] is not None else "neural=N/A"
        cll = f"classical={d['classical_mean_ll']:.3f}" if d['classical_mean_ll'] is not None else "classical=N/A"
        print(f"    {date:12s} ({d['n_events']:2d} events): {nll}  {cll}")

    # Save results
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "eval_holdout.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_dir / 'eval_holdout.json'}")

    # Plots
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_comparison_by_city(combined, plots_dir)
    plot_comparison_by_date(combined, plots_dir)
    plot_scatter_comparison(combined, plots_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
