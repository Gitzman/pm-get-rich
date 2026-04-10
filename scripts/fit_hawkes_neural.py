"""Neural Hawkes Process (continuous-time LSTM).

Implements the Mei & Eisner (2017) continuous-time LSTM Hawkes model.
Between events, the LSTM cell state decays exponentially toward a target,
producing time-varying hidden states and intensities.

Baseline to beat: classical multivariate Hawkes held-out LL = -5.473

Usage:
    uv run python scripts/fit_hawkes_neural.py [--hidden 64] [--epochs 200] [--gpu 1]
"""

from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# Data loading (same logic as classical baseline for consistency)
# ---------------------------------------------------------------------------


def load_and_prepare(
    parquet_path: Path,
    n_top_wallets: int = 50,
    min_events_per_pair: int = 10,
    max_dims: int = 50,
) -> tuple[
    np.ndarray,  # times (float64)
    np.ndarray,  # dims (int32)
    list[tuple[str, str]],  # dim labels
    int,  # train split index
    float,  # T_train
    float,  # T_all
]:
    df = pl.read_parquet(parquet_path)
    top_wallets = (
        df.group_by("actor")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(n_top_wallets)["actor"]
        .to_list()
    )
    df = df.filter(pl.col("actor").is_in(top_wallets)).sort("seq")

    pairs = (
        df.group_by("actor", "suit")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_events_per_pair)
        .sort("n", descending=True)
        .head(max_dims)
    )

    dim_map: dict[tuple[str, str], int] = {}
    dim_labels: list[tuple[str, str]] = []
    for row in pairs.iter_rows(named=True):
        key = (row["actor"], row["suit"])
        dim_map[key] = len(dim_labels)
        dim_labels.append((row["actor"][:10], row["suit"]))

    times_list: list[float] = []
    dims_list: list[int] = []
    for row in df.sort("seq").iter_rows(named=True):
        key = (row["actor"], row["suit"])
        if key in dim_map:
            times_list.append(float(row["seq"]))
            dims_list.append(dim_map[key])

    times = np.array(times_list, dtype=np.float64)
    dims = np.array(dims_list, dtype=np.int32)

    N = len(times)
    split = int(0.8 * N)
    T_train = times[split - 1] + 1.0
    T_all = times[-1] + 1.0

    return times, dims, dim_labels, split, T_train, T_all


# ---------------------------------------------------------------------------
# Continuous-Time LSTM Hawkes Model (Mei & Eisner 2017)
# Optimized: single combined linear for all gates, no batch dim
# ---------------------------------------------------------------------------


class NeuralHawkes(nn.Module):
    """Neural Hawkes Process with CT-LSTM.

    All internal computation uses 1D tensors (hidden_dim,) — no batch dimension.
    Gates are computed with a single fused linear layer for efficiency.
    """

    def __init__(self, n_dims: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.n_dims = n_dims
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(n_dims + 1, embed_dim)
        self.bos_idx = n_dims

        # Single fused linear: [x, h] -> [i, f, o, z, c_bar, delta] (6 * hidden_dim)
        self.gates = nn.Linear(embed_dim + hidden_dim, 6 * hidden_dim)

        # Intensity output
        self.intensity = nn.Linear(hidden_dim, n_dims)

    def _init_state(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Returns (h, c, c_bar, delta, o) all shape (hidden_dim,)."""
        h = torch.zeros(self.hidden_dim, device=device)
        c = torch.zeros(self.hidden_dim, device=device)
        c_bar = torch.zeros(self.hidden_dim, device=device)
        delta = torch.ones(self.hidden_dim, device=device) * 0.1
        o = torch.ones(self.hidden_dim, device=device) * 0.5
        return h, c, c_bar, delta, o

    def _cell_update(
        self, x: torch.Tensor, h: torch.Tensor, c_decayed: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """CT-LSTM cell update. All inputs/outputs are 1D (hidden_dim,).

        Returns (h_new, c_new, c_bar_new, delta_new, o_new).
        """
        xh = torch.cat([x, h])  # (embed_dim + hidden_dim,)
        gates = self.gates(xh)  # (6 * hidden_dim,)

        i, f, o, z, cb, d = gates.chunk(6)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        z = torch.tanh(z)
        c_bar = torch.tanh(cb)
        delta = F.softplus(d)

        c = f * c_decayed + i * z
        h = o * torch.tanh(c)

        return h, c, c_bar, delta, o

    def _decay_state(
        self, c: torch.Tensor, c_bar: torch.Tensor,
        delta: torch.Tensor, o: torch.Tensor, dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decay cell state and compute hidden state at time offset dt.

        Returns (h_at_t, c_at_t).
        """
        decay = torch.exp(-delta * dt)
        c_t = c_bar + (c - c_bar) * decay
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def _compute_intensity(self, h: torch.Tensor) -> torch.Tensor:
        """Compute intensity from hidden state. Input (hidden_dim,) or (M, hidden_dim)."""
        return F.softplus(self.intensity(h))

    def forward_sequence(
        self,
        times: torch.Tensor,   # (N,)
        dims: torch.Tensor,    # (N,) long
        T_end: float,
        n_mc: int = 20,
        init_state: tuple | None = None,  # carry state from previous chunk
    ) -> tuple[torch.Tensor, int, tuple]:
        """Compute log-likelihood for event sequence.

        Returns (log_likelihood, n_events, final_state).
        """
        device = times.device
        N = len(times)

        if init_state is not None:
            h, c, c_bar, delta, o, prev_t = init_state
        else:
            h, c, c_bar, delta, o = self._init_state(device)
            bos_emb = self.embedding(torch.tensor(self.bos_idx, device=device))
            h, c, c_bar, delta, o = self._cell_update(bos_emb, h, c)
            prev_t = torch.tensor(0.0, device=device)

        log_likelihood = torch.tensor(0.0, device=device)
        integral = torch.tensor(0.0, device=device)

        for n in range(N):
            t_n = times[n]
            dt = (t_n - prev_t).clamp(min=0.0)

            # MC integral over [prev_t, t_n]
            if dt > 0:
                u = torch.rand(n_mc, device=device)
                sample_dts = u * dt
                decay = torch.exp(-delta.unsqueeze(0) * sample_dts.unsqueeze(1))
                c_samples = c_bar.unsqueeze(0) + (c - c_bar).unsqueeze(0) * decay
                h_samples = o.unsqueeze(0) * torch.tanh(c_samples)
                lam_samples = self._compute_intensity(h_samples)
                integral = integral + dt * lam_samples.sum(dim=1).mean()

            # Decay to event time
            h_t, c_t = self._decay_state(c, c_bar, delta, o, dt)
            lam = self._compute_intensity(h_t)

            log_likelihood = log_likelihood + torch.log(lam[dims[n]].clamp(min=1e-8))

            # Update cell with event
            x_n = self.embedding(dims[n])
            h, c, c_bar, delta, o = self._cell_update(x_n, h_t, c_t)
            prev_t = t_n

        # Tail integral [t_N, T_end]
        dt_tail = T_end - prev_t
        if dt_tail > 0:
            u = torch.rand(n_mc, device=device)
            sample_dts = u * dt_tail
            decay = torch.exp(-delta.unsqueeze(0) * sample_dts.unsqueeze(1))
            c_samples = c_bar.unsqueeze(0) + (c - c_bar).unsqueeze(0) * decay
            h_samples = o.unsqueeze(0) * torch.tanh(c_samples)
            lam_samples = self._compute_intensity(h_samples)
            integral = integral + dt_tail * lam_samples.sum(dim=1).mean()

        final_state = (h, c, c_bar, delta, o, prev_t)
        return log_likelihood - integral, N, final_state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: NeuralHawkes,
    times_train: np.ndarray,
    dims_train: np.ndarray,
    T_train: float,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    n_mc: int = 20,
    chunk_size: int = 512,
    patience: int = 30,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    t_tensor = torch.tensor(times_train, dtype=torch.float32, device=device)
    d_tensor = torch.tensor(dims_train, dtype=torch.long, device=device)
    N = len(times_train)

    # Normalize by mean inter-event gap for numerical stability
    # This keeps intensities O(1) while preserving relative timing structure
    dt_mean = float(np.diff(times_train).mean()) if N > 1 else 1.0
    t_scale = dt_mean
    t_tensor = t_tensor / t_scale
    T_norm = T_train / t_scale

    losses = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_ll = 0.0
        epoch_n = 0
        carry_state = None  # carry LSTM state across chunks

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            t_chunk = t_tensor[start:end]
            d_chunk = d_tensor[start:end]
            T_chunk = float(t_tensor[end]) if end < N else T_norm

            optimizer.zero_grad()
            ll, n_events, final_state = model.forward_sequence(
                t_chunk, d_chunk, T_chunk, n_mc=n_mc, init_state=carry_state
            )
            nll = -ll / n_events
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Detach state for next chunk (TBPTT)
            carry_state = tuple(s.detach() if isinstance(s, torch.Tensor) else s
                                for s in final_state)

            epoch_ll += float(ll)
            epoch_n += n_events

        avg_nll = -epoch_ll / epoch_n
        losses.append(avg_nll)
        scheduler.step(avg_nll)

        if avg_nll < best_loss:
            best_loss = avg_nll
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}/{epochs}: NLL/event={avg_nll:.4f}  "
                  f"best={best_loss:.4f} (ep {best_epoch})  lr={lr_now:.2e}")

        if epoch - best_epoch > patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"  Restored best model from epoch {best_epoch} (NLL/event={best_loss:.4f})")
    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_held_out(
    model: NeuralHawkes,
    times_train: np.ndarray,
    dims_train: np.ndarray,
    times_test: np.ndarray,
    dims_test: np.ndarray,
    T_train: float,
    T_all: float,
    device: torch.device,
    n_mc: int = 100,
) -> float:
    """Held-out average log-likelihood. Runs through train to build state, evaluates on test."""
    model.eval()

    times_all = np.concatenate([times_train, times_test])
    dims_all = np.concatenate([dims_train, dims_test])

    t_tensor = torch.tensor(times_all, dtype=torch.float32, device=device)
    d_tensor = torch.tensor(dims_all, dtype=torch.long, device=device)

    # Same normalization as training: mean inter-event gap
    dt_mean = float(np.diff(times_train).mean()) if len(times_train) > 1 else 1.0
    t_scale = dt_mean
    t_tensor = t_tensor / t_scale
    T_all_norm = T_all / t_scale

    N_train = len(times_train)
    N_test = len(times_test)
    N = N_train + N_test

    with torch.no_grad():
        h, c, c_bar, delta, o = model._init_state(device)
        bos_emb = model.embedding(torch.tensor(model.bos_idx, device=device))
        h, c, c_bar, delta, o = model._cell_update(bos_emb, h, c)

        ll_test = 0.0
        integral_test = 0.0
        prev_t = torch.tensor(0.0, device=device)

        for n in range(N):
            t_n = t_tensor[n]
            dt = (t_n - prev_t).clamp(min=0.0)

            # MC integral (only for test portion)
            if n >= N_train and dt > 0:
                u = torch.rand(n_mc, device=device)
                sample_dts = u * dt
                decay = torch.exp(-delta.unsqueeze(0) * sample_dts.unsqueeze(1))
                c_samples = c_bar.unsqueeze(0) + (c - c_bar).unsqueeze(0) * decay
                h_samples = o.unsqueeze(0) * torch.tanh(c_samples)
                lam_samples = model._compute_intensity(h_samples)
                integral_test += float(dt * lam_samples.sum(dim=1).mean())

            # Decay to event
            h_t, c_t = model._decay_state(c, c_bar, delta, o, dt)
            lam = model._compute_intensity(h_t)

            if n >= N_train:
                ll_test += float(torch.log(lam[d_tensor[n]].clamp(min=1e-8)))

            # Update
            x_n = model.embedding(d_tensor[n])
            h, c, c_bar, delta, o = model._cell_update(x_n, h_t, c_t)
            prev_t = t_n

        # Tail integral
        dt_tail = T_all_norm - prev_t
        if dt_tail > 0:
            u = torch.rand(n_mc, device=device)
            sample_dts = u * dt_tail
            decay = torch.exp(-delta.unsqueeze(0) * sample_dts.unsqueeze(1))
            c_samples = c_bar.unsqueeze(0) + (c - c_bar).unsqueeze(0) * decay
            h_samples = o.unsqueeze(0) * torch.tanh(c_samples)
            lam_samples = model._compute_intensity(h_samples)
            integral_test += float(dt_tail * lam_samples.sum(dim=1).mean())

        ll_test -= integral_test

        # Correct for time normalization: lambda_orig = lambda_norm / t_scale
        # log(lambda_orig) = log(lambda_norm) - log(t_scale) per event
        # integral is invariant under change of variables
        log_t_scale = float(np.log(t_scale))
        ll_test -= N_test * log_t_scale

        return ll_test / N_test


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def compute_intensity_over_time(
    model: NeuralHawkes,
    times: np.ndarray,
    dims: np.ndarray,
    eval_times: np.ndarray,
    target_dims: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    model.eval()
    dt_mean = float(np.diff(times).mean()) if len(times) > 1 else 1.0
    t_scale = dt_mean
    t_tensor = torch.tensor(times / t_scale, dtype=torch.float32, device=device)
    d_tensor = torch.tensor(dims, dtype=torch.long, device=device)
    eval_t_norm = eval_times / t_scale
    N = len(times)

    result = {d: np.zeros(len(eval_times)) for d in target_dims}

    with torch.no_grad():
        h, c, c_bar, delta, o = model._init_state(device)
        bos_emb = model.embedding(torch.tensor(model.bos_idx, device=device))
        h, c, c_bar, delta, o = model._cell_update(bos_emb, h, c)

        event_idx = 0
        prev_t = torch.tensor(0.0, device=device)

        for ei, et in enumerate(eval_t_norm):
            et_t = torch.tensor(et, dtype=torch.float32, device=device)

            while event_idx < N and t_tensor[event_idx] <= et_t:
                t_n = t_tensor[event_idx]
                dt = (t_n - prev_t).clamp(min=0.0)
                h_t, c_t = model._decay_state(c, c_bar, delta, o, dt)
                x_n = model.embedding(d_tensor[event_idx])
                h, c, c_bar, delta, o = model._cell_update(x_n, h_t, c_t)
                prev_t = t_n
                event_idx += 1

            dt_eval = (et_t - prev_t).clamp(min=0.0)
            h_t, _ = model._decay_state(c, c_bar, delta, o, dt_eval)
            lam = model._compute_intensity(h_t)

            for d in target_dims:
                result[d][ei] = float(lam[d])

    return result


def plot_intensities(
    model: NeuralHawkes,
    times_train: np.ndarray,
    dims_train: np.ndarray,
    dim_labels: list[tuple[str, str]],
    target_dims: list[int],
    output_dir: Path,
    device: torch.device,
    n_eval_points: int = 500,
) -> None:
    t_min, t_max = times_train[0], times_train[-1]
    eval_times = np.linspace(t_min, t_max, n_eval_points)

    intensities = compute_intensity_over_time(
        model, times_train, dims_train, eval_times, target_dims, device
    )

    fig, axes = plt.subplots(len(target_dims), 1, figsize=(14, 3.5 * len(target_dims)),
                             sharex=True)
    if len(target_dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, target_dims):
        label = f"{dim_labels[dim][0]}_{dim_labels[dim][1]}"
        ax.plot(eval_times, intensities[dim], color="steelblue", linewidth=0.8,
                label="Neural intensity")
        event_mask = dims_train == dim
        event_times = times_train[event_mask]
        ax.scatter(event_times, np.zeros_like(event_times), marker="|", color="red",
                   alpha=0.5, s=30, label="Events", zorder=5)
        ax.set_ylabel(f"$\\lambda_{{{dim}}}(t)$")
        ax.set_title(f"Dim {dim}: {label}")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(t_min, t_max)

    axes[-1].set_xlabel("seq (time)")
    plt.suptitle("Neural Hawkes Intensity Functions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "intensity_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'intensity_functions.png'}")


def plot_training_curve(losses: list[float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, color="steelblue", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL / event")
    ax.set_title("Neural Hawkes Training Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'training_curve.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Hawkes (CT-LSTM)")
    parser.add_argument("--parquet", type=Path, default=Path("data/events/295980/events.parquet"))
    parser.add_argument("--max-dims", type=int, default=50)
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--top-wallets", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/reports/hawkes_neural"))
    args = parser.parse_args()

    # Load event metadata if available
    meta_path = args.parquet.parent / "_meta.json"
    event_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            event_meta = json.load(f)
    event_id = event_meta.get("event_id", args.parquet.parent.name)
    event_title = event_meta.get("event_title", "unknown")

    print("=" * 60)
    print("Neural Hawkes Process (CT-LSTM)")
    print(f"Event: {event_title} ({event_id})")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("\nWARNING: CUDA not available, using CPU")

    # 1. Load data
    print("\n[1/5] Loading and preparing data...")
    times, dims, dim_labels, split, T_train, T_all = load_and_prepare(
        args.parquet, n_top_wallets=args.top_wallets,
        min_events_per_pair=args.min_events, max_dims=args.max_dims,
    )
    D = len(dim_labels)
    N = len(times)
    N_train = split
    N_test = N - split

    print(f"  Dimensions (D): {D}")
    print(f"  Total events: {N}")
    print(f"  Training events: {N_train} (80%)")
    print(f"  Test events: {N_test} (20%)")
    print(f"  Time range: seq 0 to {times[-1]:.0f}")

    train_times = times[:split]
    train_dims = dims[:split]
    test_times = times[split:]
    test_dims = dims[split:]

    # 2. Build model
    print(f"\n[2/5] Building model (hidden={args.hidden}, embed={args.embed_dim})...")
    model = NeuralHawkes(n_dims=D, hidden_dim=args.hidden, embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # 3. Train
    print(f"\n[3/5] Training (epochs={args.epochs}, lr={args.lr}, chunks={args.chunk_size})...")
    t0 = time.time()
    losses = train_model(
        model, train_times, train_dims, T_train, device,
        epochs=args.epochs, lr=args.lr, n_mc=args.mc_samples,
        chunk_size=args.chunk_size, patience=args.patience,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    vram_used = 0.0
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  Peak VRAM usage: {vram_used:.1f} MB")

    # 4. Evaluate
    print("\n[4/5] Evaluating held-out log-likelihood...")
    ll_estimates = []
    for run in range(5):
        ll = eval_held_out(
            model, train_times, train_dims, test_times, test_dims,
            T_train, T_all, device, n_mc=200,
        )
        ll_estimates.append(ll)
        print(f"  Run {run+1}: held-out avg LL = {ll:.4f}")

    held_out_ll = float(np.mean(ll_estimates))
    held_out_std = float(np.std(ll_estimates))
    print(f"\n  Held-out avg LL: {held_out_ll:.4f} +/- {held_out_std:.4f}")

    # Try to load classical baseline for comparison
    classical_ll = None
    classical_results_path = args.out / "results.json"
    # Check sibling directory or same directory for classical results
    for candidate in [args.out.parent / args.out.name / "results.json",
                      args.parquet.parent.parent.parent / "reports" / "generalization" / event_id / "results.json",
                      Path("data/reports/hawkes_classical/results.json")]:
        if candidate.exists():
            try:
                with open(candidate) as f:
                    cr = json.load(f)
                if cr.get("model") == "multivariate_exponential_hawkes":
                    classical_ll = cr["metrics"]["held_out_avg_log_likelihood"]
                    break
            except Exception:
                pass
    if classical_ll is None:
        classical_ll = -5.473  # fallback default

    print(f"  Classical baseline: {classical_ll:.4f}")
    if held_out_ll > classical_ll:
        print(f"  >>> BEATS classical baseline by {held_out_ll - classical_ll:.4f} <<<")
    else:
        print(f"  >>> Below classical baseline by {classical_ll - held_out_ll:.4f} <<<")

    # 5. Plots
    print("\n[5/5] Generating visualizations...")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curve(losses, out_dir)

    dim_counts = np.bincount(train_dims, minlength=D)
    top_active = np.argsort(dim_counts)[::-1][:5].tolist()
    print(f"  Plotting intensities for dims: {top_active}")
    plot_intensities(model, train_times, train_dims, dim_labels, top_active,
                     out_dir, device)

    results = {
        "event_id": event_id,
        "event_title": event_title,
        "model": "neural_hawkes_ct_lstm",
        "architecture": "Mei & Eisner (2017) continuous-time LSTM",
        "hidden_dim": args.hidden,
        "embed_dim": args.embed_dim,
        "n_parameters": n_params,
        "dimensions": D,
        "n_train": N_train,
        "n_test": N_test,
        "training": {
            "epochs": len(losses),
            "lr": args.lr,
            "chunk_size": args.chunk_size,
            "mc_samples": args.mc_samples,
            "patience": args.patience,
            "final_train_nll_per_event": float(losses[-1]),
            "best_train_nll_per_event": float(min(losses)),
            "training_time_seconds": train_time,
        },
        "metrics": {
            "held_out_avg_log_likelihood": held_out_ll,
            "held_out_ll_std": held_out_std,
            "classical_baseline_ll": classical_ll,
            "beats_baseline": bool(held_out_ll > classical_ll),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu",
            "peak_vram_mb": vram_used,
        },
        "dimension_labels": [
            {"idx": i, "wallet": w, "suit": s}
            for i, (w, s) in enumerate(dim_labels)
        ],
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    ckpt_path = out_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Model:                       CT-LSTM Neural Hawkes")
    print(f"  Dimensions:                  {D}")
    print(f"  Parameters:                  {n_params:,}")
    print(f"  Training epochs:             {len(losses)}")
    print(f"  Best train NLL/event:        {min(losses):.4f}")
    print(f"  Held-out avg LL:             {held_out_ll:.4f} +/- {held_out_std:.4f}")
    print(f"  Classical baseline LL:       {classical_ll:.4f}")
    print(f"  Beats baseline:              {'YES' if held_out_ll > classical_ll else 'NO'}")
    if torch.cuda.is_available():
        print(f"  GPU:                         {torch.cuda.get_device_name(args.gpu)}")
        print(f"  Peak VRAM:                   {vram_used:.1f} MB")
    print(f"  Reports:                     {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
