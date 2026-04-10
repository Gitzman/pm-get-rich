"""Generate trading signals from Neural Hawkes TPP intensity predictions.

Convoy 2 deliverable (pm-kxg.2).

For each qualifying market:
  1. Load events using real timestamps (seconds since first event)
  2. Train Neural Hawkes (CT-LSTM) on first 80% of events
  3. At each test-set event, predict total intensity Δt seconds ahead
  4. Apply per-market percentile thresholds to identify signal points

Sweeps: Δt ∈ {15, 30, 60, 120}s, threshold ∈ {top 1%, 5%, 10%}

Output: data/signals.parquet

Usage:
    uv run python scripts/generate_signals.py [--min-events 1000] [--max-markets 0] [--gpu 0]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import torch

from scripts.fit_hawkes_neural import NeuralHawkes, train_model


def _print(*args, **kwargs):
    """Print with flush for unbuffered output in batch jobs."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

DELTA_TS = [15, 30, 60, 120]  # seconds
THRESHOLD_PCTS = [1, 5, 10]   # top X%


# ---------------------------------------------------------------------------
# Data loading (real timestamps)
# ---------------------------------------------------------------------------


def load_events_realtime(
    parquet_path: Path,
    n_top_wallets: int = 50,
    min_events_per_pair: int = 10,
    max_dims: int = 50,
) -> dict:
    """Load events using real timestamps instead of seq ordinals.

    Returns dict with: times_s, dims, dim_labels, split, T_train, T_all,
    prices, times_ms, suits.
    """
    df = pl.read_parquet(parquet_path)

    top_wallets = (
        df.group_by("actor")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(n_top_wallets)["actor"]
        .to_list()
    )
    df = df.filter(pl.col("actor").is_in(top_wallets)).sort("timestamp_ms")

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

    times_ms_list: list[float] = []
    dims_list: list[int] = []
    prices_list: list[float] = []
    suits_list: list[str] = []

    for row in df.sort("timestamp_ms").iter_rows(named=True):
        key = (row["actor"], row["suit"])
        if key in dim_map:
            times_ms_list.append(float(row["timestamp_ms"]))
            dims_list.append(dim_map[key])
            prices_list.append(float(row["price"]))
            suits_list.append(row["suit"])

    times_ms = np.array(times_ms_list, dtype=np.float64)
    times_s = (times_ms - times_ms[0]) / 1000.0
    dims = np.array(dims_list, dtype=np.int32)
    prices = np.array(prices_list, dtype=np.float64)

    N = len(times_s)
    split = int(0.8 * N)
    T_train = times_s[split - 1] + 1.0
    T_all = times_s[-1] + 1.0

    return {
        "times_s": times_s,
        "dims": dims,
        "dim_labels": dim_labels,
        "split": split,
        "T_train": T_train,
        "T_all": T_all,
        "prices": prices,
        "times_ms": times_ms,
        "suits": np.array(suits_list),
    }


# ---------------------------------------------------------------------------
# Forward intensity prediction (no look-ahead bias)
# ---------------------------------------------------------------------------


def compute_forward_intensities(
    model: NeuralHawkes,
    times: np.ndarray,
    dims: np.ndarray,
    eval_indices: np.ndarray,
    delta_ts: list[int],
    t_scale: float,
    device: torch.device,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Predict intensity Δt seconds ahead at each eval event.

    At each eval index i:
      1. All events 0..i have been processed (state built up)
      2. Decay state forward by Δt seconds (no future events)
      3. Return total intensity, argmax dim, and max dim intensity

    Uses t_scale from training (mean inter-event gap of train set) to
    match the normalization used during model training.

    Returns: {delta_t: (total_intensity, max_dim_idx, max_dim_intensity)}
    """
    model.eval()

    t_tensor = torch.tensor(times / t_scale, dtype=torch.float32, device=device)
    d_tensor = torch.tensor(dims, dtype=torch.long, device=device)

    n_eval = len(eval_indices)
    N = len(times)

    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dt in delta_ts:
        result[dt] = (
            np.zeros(n_eval),
            np.zeros(n_eval, dtype=np.int32),
            np.zeros(n_eval),
        )

    with torch.no_grad():
        h, c, c_bar, delta, o = model._init_state(device)
        bos_emb = model.embedding(torch.tensor(model.bos_idx, device=device))
        h, c, c_bar, delta, o = model._cell_update(bos_emb, h, c)

        prev_t = torch.tensor(0.0, device=device)
        eval_pos = 0

        for n in range(N):
            t_n = t_tensor[n]
            dt_event = (t_n - prev_t).clamp(min=0.0)

            # Decay to event time and update with event
            h_t, c_t = model._decay_state(c, c_bar, delta, o, dt_event)
            x_n = model.embedding(d_tensor[n])
            h, c, c_bar, delta, o = model._cell_update(x_n, h_t, c_t)
            prev_t = t_n

            # Evaluate forward intensities if this is an eval point
            if eval_pos < n_eval and n == eval_indices[eval_pos]:
                for dt_fwd in delta_ts:
                    dt_norm = torch.tensor(
                        dt_fwd / t_scale, dtype=torch.float32, device=device
                    )
                    h_fwd, _ = model._decay_state(c, c_bar, delta, o, dt_norm)
                    lam = model._compute_intensity(h_fwd)  # (D,)

                    total = float(lam.sum())
                    max_idx = int(lam.argmax())
                    max_val = float(lam[max_idx])

                    result[dt_fwd][0][eval_pos] = total
                    result[dt_fwd][1][eval_pos] = max_idx
                    result[dt_fwd][2][eval_pos] = max_val

                eval_pos += 1

    return result


# ---------------------------------------------------------------------------
# Per-market processing
# ---------------------------------------------------------------------------


def process_market(
    event_id: str,
    parquet_path: Path,
    device: torch.device,
    hidden_dim: int = 32,
    embed_dim: int = 16,
    epochs: int = 50,
    patience: int = 15,
    chunk_size: int = 256,
) -> pl.DataFrame | None:
    """Train model and generate signals for one market.

    Returns a DataFrame of signal rows, or None if the market is skipped.
    """
    try:
        data = load_events_realtime(parquet_path)
    except Exception as e:
        _print(f"  SKIP {event_id}: load failed: {e}")
        return None

    times_s = data["times_s"]
    dims = data["dims"]
    dim_labels = data["dim_labels"]
    split = data["split"]
    T_train = data["T_train"]
    prices = data["prices"]
    times_ms = data["times_ms"]

    D = len(dim_labels)
    N = len(times_s)
    N_test = N - split

    if D < 2:
        _print(f"  SKIP {event_id}: only {D} dimension(s)")
        return None

    if N_test < 20:
        _print(f"  SKIP {event_id}: only {N_test} test events")
        return None

    # Train (suppress per-epoch output for batch mode)
    model = NeuralHawkes(
        n_dims=D, hidden_dim=hidden_dim, embed_dim=embed_dim
    ).to(device)

    train_times = times_s[:split]
    train_dims = dims[:split]

    # Compute t_scale the same way train_model does internally
    t_scale = float(np.diff(train_times).mean()) if len(train_times) > 1 else 1.0

    # Suppress per-epoch output in batch mode
    with redirect_stdout(io.StringIO()):
        train_model(
            model,
            train_times,
            train_dims,
            T_train,
            device,
            epochs=epochs,
            patience=patience,
            chunk_size=chunk_size,
            lr=1e-3,
            n_mc=10,
        )

    # Predict forward intensities on test events
    eval_indices = np.arange(split, N)
    intensities = compute_forward_intensities(
        model, times_s, dims, eval_indices, DELTA_TS, t_scale, device
    )

    # Build signal rows
    rows: list[dict] = []

    for dt_fwd in DELTA_TS:
        total_int, max_dim_idx, max_dim_int = intensities[dt_fwd]

        for pct in THRESHOLD_PCTS:
            threshold = float(np.percentile(total_int, 100 - pct))
            if threshold <= 0:
                continue

            signal_mask = total_int >= threshold

            for i in np.where(signal_mask)[0]:
                event_idx = split + i
                dim_idx = int(max_dim_idx[i])
                suit_label = dim_labels[dim_idx][1] if dim_idx < D else "?"

                # Forward price: last trade at or before t + Δt
                target_t = times_s[event_idx] + dt_fwd
                fwd_idx = int(np.searchsorted(times_s, target_t, side="right")) - 1
                fwd_idx = max(0, min(fwd_idx, N - 1))
                fwd_price = float(prices[fwd_idx])

                rows.append(
                    {
                        "event_id": event_id,
                        "timestamp_ms": int(times_ms[event_idx]),
                        "delta_t_s": dt_fwd,
                        "threshold_pct": pct,
                        "threshold_value": threshold,
                        "intensity_total": float(total_int[i]),
                        "intensity_ratio": float(total_int[i] / threshold),
                        "max_dim_suit": suit_label,
                        "max_dim_intensity": float(max_dim_int[i]),
                        "price_at_signal": float(prices[event_idx]),
                        "price_forward": fwd_price,
                        "price_change": fwd_price - float(prices[event_idx]),
                        "n_dims": D,
                        "n_test_events": N_test,
                    }
                )

    if not rows:
        return None

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------


def discover_markets(
    events_dir: Path,
    min_events: int,
    max_markets: int,
) -> list[tuple[str, Path, int]]:
    """Find qualifying markets sorted by event count (descending).

    Returns list of (event_id, parquet_path, n_events).
    """
    markets: list[tuple[str, Path, int]] = []

    for d in sorted(events_dir.iterdir()):
        pq = d / "events.parquet"
        if not pq.exists():
            continue
        # Quick row count via metadata (avoids full read)
        try:
            df = pl.scan_parquet(pq).select(pl.len()).collect()
            n = df.item()
        except Exception:
            continue
        if n >= min_events:
            markets.append((d.name, pq, n))

    markets.sort(key=lambda x: x[2], reverse=True)

    if max_markets > 0:
        markets = markets[:max_markets]

    return markets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TPP trading signals (Convoy 2)"
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=Path("data/events"),
        help="Directory containing per-market event directories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/signals.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=1000,
        help="Minimum events per market to include",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help="Max markets to process (0 = all qualifying)",
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    _print("=" * 60)
    _print("TPP Signal Generator (Convoy 2)")
    _print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        _print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        _print("WARNING: CUDA not available, using CPU (will be slow)")

    # Discover markets
    _print(f"\nDiscovering markets (min_events={args.min_events})...")
    markets = discover_markets(args.events_dir, args.min_events, args.max_markets)
    _print(f"Found {len(markets)} qualifying markets")

    if not markets:
        _print("No markets found. Check --events-dir and --min-events.")
        sys.exit(1)

    total_events = sum(n for _, _, n in markets)
    _print(f"Total events across all markets: {total_events:,}")
    _print(f"Sweep: Δt={DELTA_TS}s × threshold=top {THRESHOLD_PCTS}%")
    _print(f"Model: hidden={args.hidden}, embed={args.embed_dim}, "
           f"epochs={args.epochs}, patience={args.patience}")

    # Process markets
    all_frames: list[pl.DataFrame] = []
    n_signals_total = 0
    t_start = time.time()

    for idx, (event_id, pq_path, n_events) in enumerate(markets):
        t0 = time.time()

        df = process_market(
            event_id,
            pq_path,
            device,
            hidden_dim=args.hidden,
            embed_dim=args.embed_dim,
            epochs=args.epochs,
            patience=args.patience,
            chunk_size=args.chunk_size,
        )

        elapsed = time.time() - t0

        if df is not None:
            n_signals = len(df)
            n_signals_total += n_signals
            all_frames.append(df)
            _print(f"[{idx + 1}/{len(markets)}] {event_id} "
                   f"({n_events:,} events) => {n_signals:,} signals "
                   f"in {elapsed:.1f}s")
        else:
            _print(f"[{idx + 1}/{len(markets)}] {event_id} "
                   f"({n_events:,} events) => skip in {elapsed:.1f}s")

        # Clear GPU cache between markets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_start

    if not all_frames:
        _print("\nNo signals generated across any market.")
        sys.exit(1)

    # Concatenate and write
    result = pl.concat(all_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.out)

    # Summary
    _print("\n" + "=" * 60)
    _print("SIGNAL GENERATION COMPLETE")
    _print("=" * 60)
    _print(f"  Markets processed: {len(markets)}")
    _print(f"  Markets with signals: {len(all_frames)}")
    _print(f"  Total signals: {n_signals_total:,}")
    _print(f"  Output: {args.out}")
    _print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")

    # Per-config breakdown
    _print(f"\n  Signal counts by (Δt, threshold):")
    for dt in DELTA_TS:
        for pct in THRESHOLD_PCTS:
            count = result.filter(
                (pl.col("delta_t_s") == dt) & (pl.col("threshold_pct") == pct)
            ).height
            _print(f"    Δt={dt:3d}s, top {pct:2d}%: {count:,} signals")

    # Price change stats
    _print(f"\n  Forward price change by Δt:")
    for dt in DELTA_TS:
        sub = result.filter(pl.col("delta_t_s") == dt)["price_change"]
        if sub.len() > 0:
            _print(
                f"    Δt={dt:3d}s: mean={sub.mean():.6f}, "
                f"std={sub.std():.6f}, "
                f"median={sub.median():.6f}"
            )

    _print("=" * 60)


if __name__ == "__main__":
    main()
