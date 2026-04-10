"""Classical multivariate exponential Hawkes baseline.

Fits a multivariate Hawkes process with exponential kernels on the top 50
(wallet, suit) pairs by event count. Uses seq as the time axis since median
inter-event gap is 0.0s due to second-resolution timestamps.

Train on first 80% of seq-ordered events, hold out last 20%.

Usage:
    uv run python scripts/fit_hawkes_classical.py [--max-dims 50] [--min-events 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------


def load_and_prepare(
    parquet_path: Path,
    n_top_wallets: int = 50,
    min_events_per_pair: int = 10,
    max_dims: int = 50,
) -> tuple[
    np.ndarray,  # times (float64)
    np.ndarray,  # dims (int32)
    list[tuple[str, str]],  # dim labels (wallet_short, suit)
    int,  # train split index
    float,  # T_train (end of training window)
    float,  # T_all (end of full window)
    pl.DataFrame,  # filtered dataframe
]:
    """Load parquet, filter to top wallets, create dimension mapping."""
    df = pl.read_parquet(parquet_path)

    # Top N wallets by trade count (actor = taker)
    top_wallets = (
        df.group_by("actor")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(n_top_wallets)["actor"]
        .to_list()
    )
    df = df.filter(pl.col("actor").is_in(top_wallets)).sort("seq")

    # (wallet, suit) pairs with enough events
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

    # Build event arrays
    times_list: list[float] = []
    dims_list: list[int] = []
    for row in df.sort("seq").iter_rows(named=True):
        key = (row["actor"], row["suit"])
        if key in dim_map:
            times_list.append(float(row["seq"]))
            dims_list.append(dim_map[key])

    times = np.array(times_list, dtype=np.float64)
    dims = np.array(dims_list, dtype=np.int32)

    # 80/20 train/test split
    N = len(times)
    split = int(0.8 * N)
    T_train = times[split - 1] + 1.0  # end of training window
    T_all = times[-1] + 1.0

    return times, dims, dim_labels, split, T_train, T_all, df


# ---------------------------------------------------------------------------
# Hawkes log-likelihood (exponential kernel, shared beta)
# ---------------------------------------------------------------------------


def neg_log_likelihood(
    params: np.ndarray,
    times: np.ndarray,
    dims: np.ndarray,
    D: int,
    T: float,
    l1_penalty: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Negative log-likelihood and gradient for multivariate exponential Hawkes.

    Parameters are packed as: [mu (D), alpha (D*D), log_beta (1)]
    Kernel: g_{ij}(dt) = alpha_{ij} * beta * exp(-beta * dt)

    Returns (neg_ll, gradient).
    """
    mu = params[:D]
    alpha = params[D : D + D * D].reshape(D, D)
    log_beta = params[D + D * D]
    beta = np.exp(log_beta)

    N = len(times)

    # Recursive auxiliary variables: A[j] = sum_{k<n, d_k=j} exp(-beta * (t_n - t_k))
    A = np.zeros(D, dtype=np.float64)
    # For gradient of mu and alpha
    grad_mu = np.zeros(D, dtype=np.float64)
    grad_alpha = np.zeros((D, D), dtype=np.float64)
    grad_log_beta = 0.0

    ll = 0.0

    # Forward pass: compute log-likelihood and gradients
    prev_t = times[0]

    for n in range(N):
        t_n = times[n]
        d_n = dims[n]

        if n > 0:
            dt = t_n - prev_t
            decay = np.exp(-beta * dt)
            # Update recursive variables
            indicator = np.zeros(D)
            indicator[dims[n - 1]] = 1.0
            A = decay * (A + indicator)

        # Intensity at event n
        lam_n = mu[d_n] + beta * np.dot(alpha[d_n], A)

        if lam_n <= 1e-15:
            lam_n = 1e-15

        ll += np.log(lam_n)

        # Gradient contributions from log(lambda) term
        inv_lam = 1.0 / lam_n
        grad_mu[d_n] += inv_lam
        grad_alpha[d_n] += beta * A * inv_lam
        # d(log lam)/d(log_beta) = d(log lam)/d(beta) * beta
        # = (alpha[d_n] . A + beta * alpha[d_n] . dA/dbeta) * inv_lam * beta
        # For simplicity, skip beta gradient in recursive form; will use grid search

        prev_t = t_n

    # Compensator: integral of lambda over [0, T]
    # For mu: mu_i * T
    # For alpha: sum_j alpha_{ij} * sum_{k: d_k=j} (1 - exp(-beta * (T - t_k)))
    compensator_alpha = np.zeros((D, D), dtype=np.float64)
    for n in range(N):
        j = dims[n]
        c = 1.0 - np.exp(-beta * (T - times[n]))
        compensator_alpha[:, j] += c

    ll -= np.sum(mu * T)
    ll -= np.sum(alpha * compensator_alpha)

    # Gradient from compensator
    grad_mu -= T
    grad_alpha -= compensator_alpha

    # L1 penalty on alpha (differentiable approximation: |x| ≈ x for x >= 0 with bounds)
    # Since alpha >= 0 (enforced by bounds), L1 = sum(alpha)
    if l1_penalty > 0:
        ll -= l1_penalty * np.sum(alpha)
        grad_alpha -= l1_penalty

    # Pack gradient (negate for minimization)
    grad = np.zeros_like(params)
    grad[:D] = -grad_mu
    grad[D : D + D * D] = -grad_alpha.ravel()
    # grad for log_beta is omitted (grid search)
    grad[D + D * D] = 0.0

    return -ll, grad


# ---------------------------------------------------------------------------
# Held-out log-likelihood evaluation
# ---------------------------------------------------------------------------


def eval_log_likelihood(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    times_train: np.ndarray,
    dims_train: np.ndarray,
    times_test: np.ndarray,
    dims_test: np.ndarray,
    T_test_end: float,
    D: int,
) -> float:
    """Compute average log-likelihood on held-out events.

    Initializes recursive state from end of training set, then evaluates
    on test events.
    """
    # Build A from training data at end of training
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

    # Evaluate on test events (continuing from training state)
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

    # Compensator on test window
    T_test_start = times_test[0]
    T_duration = T_test_end - T_test_start

    # Baseline compensator
    ll -= np.sum(mu) * T_duration  # approximate: all dims active

    # More precisely: compensator for each dim
    for i in range(D):
        ll -= mu[i] * T_duration

    # Undo the approximate subtraction above
    ll += np.sum(mu) * T_duration

    # Alpha compensator on test events
    for n in range(N_test):
        j = dims_test[n]
        c = 1.0 - np.exp(-beta * (T_test_end - times_test[n]))
        ll -= np.sum(alpha[:, j]) * c

    # Also account for training events still influencing the test window
    for n in range(len(times_train)):
        j = dims_train[n]
        c_end = 1.0 - np.exp(-beta * (T_test_end - times_train[n]))
        c_start = 1.0 - np.exp(-beta * (T_test_start - times_train[n]))
        ll -= np.sum(alpha[:, j]) * (c_end - c_start)

    return ll / N_test  # average per-event


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_hawkes(
    times: np.ndarray,
    dims: np.ndarray,
    D: int,
    T: float,
    beta_values: list[float],
    l1_penalty: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit multivariate Hawkes via grid search over beta + L-BFGS-B for mu, alpha.

    Returns (mu, alpha, beta, train_nll).
    """
    best_nll = np.inf
    best_result = None
    best_beta = None

    for beta_val in beta_values:
        log_beta = np.log(beta_val)
        print(f"  Trying beta={beta_val:.4f} (log_beta={log_beta:.3f})...", end=" ")

        # Initial params
        # mu: empirical rate per dimension
        N = len(times)
        mu_init = np.zeros(D)
        for d in range(D):
            mu_init[d] = max(np.sum(dims == d) / T, 1e-6)
        alpha_init = np.full(D * D, 0.001)

        x0 = np.concatenate([mu_init, alpha_init, [log_beta]])

        # Bounds: mu >= 1e-8, alpha >= 0, log_beta fixed
        bounds = [(1e-8, None)] * D  # mu
        bounds += [(0.0, 1.0)] * (D * D)  # alpha (capped to help stationarity)
        bounds += [(log_beta, log_beta)]  # fix log_beta

        t0 = time.time()
        result = minimize(
            neg_log_likelihood,
            x0,
            args=(times, dims, D, T, l1_penalty),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        elapsed = time.time() - t0

        print(f"nll={result.fun:.2f}, converged={result.success}, {elapsed:.1f}s")

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
            best_beta = beta_val

    # Extract best parameters
    x = best_result.x
    mu = x[:D]
    alpha = x[D : D + D * D].reshape(D, D)
    return mu, alpha, best_beta, best_nll


# ---------------------------------------------------------------------------
# Metrics & visualization
# ---------------------------------------------------------------------------


def compute_branching_ratio(alpha: np.ndarray) -> float:
    """Branching ratio = spectral radius of alpha matrix.

    Must be < 1 for stationarity.
    """
    eigenvalues = np.linalg.eigvals(alpha)
    return float(np.max(np.abs(eigenvalues)))


def plot_influence_matrix(
    alpha: np.ndarray,
    dim_labels: list[tuple[str, str]],
    output_path: Path,
    beta: float,
) -> None:
    """Plot heatmap of the influence (kernel) matrix."""
    D = alpha.shape[0]
    short_labels = [f"{w}_{s}" for w, s in dim_labels]

    fig, ax = plt.subplots(figsize=(max(12, D * 0.3), max(10, D * 0.25)))

    # Use log scale for better visibility of sparse matrix
    alpha_plot = alpha.copy()
    alpha_plot[alpha_plot < 1e-6] = 0

    im = ax.imshow(alpha_plot, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xlabel("Source dimension (j → triggers)")
    ax.set_ylabel("Target dimension (i ← triggered)")
    ax.set_title(f"Hawkes Influence Matrix α (β={beta:.4f})\nBranching ratio: {compute_branching_ratio(alpha):.4f}")

    if D <= 30:
        ax.set_xticks(range(D))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(D))
        ax.set_yticklabels(short_labels, fontsize=6)

    plt.colorbar(im, ax=ax, shrink=0.8, label="α_ij")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_baseline_intensities(
    mu: np.ndarray,
    dim_labels: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """Bar plot of fitted baseline intensities."""
    D = len(mu)
    short_labels = [f"{w}_{s}" for w, s in dim_labels]

    fig, ax = plt.subplots(figsize=(max(10, D * 0.25), 5))
    ax.bar(range(D), mu, color="steelblue", alpha=0.8)
    ax.set_xlabel("Dimension (wallet_suit)")
    ax.set_ylabel("Baseline intensity μ_i")
    ax.set_title("Fitted Baseline Intensities")

    if D <= 30:
        ax.set_xticks(range(D))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_influences(
    alpha: np.ndarray,
    dim_labels: list[tuple[str, str]],
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Plot top N strongest influence pairs."""
    D = alpha.shape[0]
    short_labels = [f"{w}_{s}" for w, s in dim_labels]

    # Get top influences
    flat_idx = np.argsort(alpha.ravel())[::-1][:top_n]
    rows, cols = np.unravel_index(flat_idx, (D, D))

    labels = [f"{short_labels[c]}→{short_labels[r]}" for r, c in zip(rows, cols)]
    values = [alpha[r, c] for r, c in zip(rows, cols)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(values) - 1, -1, -1), values, color="coral", alpha=0.8)
    ax.set_yticks(range(len(values) - 1, -1, -1))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("α_ij (influence strength)")
    ax.set_title(f"Top {top_n} Influence Pairs (source → target)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit classical Hawkes baseline")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/events/295980/events.parquet"),
        help="Path to normalized events parquet",
    )
    parser.add_argument("--max-dims", type=int, default=50, help="Max (wallet,suit) dimensions")
    parser.add_argument("--min-events", type=int, default=10, help="Min events per (wallet,suit) pair")
    parser.add_argument("--top-wallets", type=int, default=50, help="Top N wallets by trade count")
    parser.add_argument("--l1", type=float, default=0.05, help="L1 penalty on alpha")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/reports/hawkes_classical"),
        help="Output directory",
    )
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
    print("Classical Multivariate Hawkes Baseline")
    print(f"Event: {event_title} ({event_id})")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    times, dims, dim_labels, split, T_train, T_all, df = load_and_prepare(
        args.parquet,
        n_top_wallets=args.top_wallets,
        min_events_per_pair=args.min_events,
        max_dims=args.max_dims,
    )
    D = len(dim_labels)
    N = len(times)
    N_train = split
    N_test = N - split

    print(f"  Dimensions (D): {D}")
    print(f"  Total events: {N}")
    print(f"  Training events: {N_train} (80%)")
    print(f"  Test events: {N_test} (20%)")
    print(f"  Time axis: seq (0 to {times[-1]:.0f})")
    print(f"  T_train: {T_train:.0f}")

    train_times = times[:split]
    train_dims = dims[:split]
    test_times = times[split:]
    test_dims = dims[split:]

    # 2. Fit model (grid search over beta)
    print("\n[2/5] Fitting Hawkes model (grid search over β)...")
    beta_grid = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    mu, alpha, beta, train_nll = fit_hawkes(
        train_times,
        train_dims,
        D,
        T_train,
        beta_grid,
        l1_penalty=args.l1,
    )
    print(f"\n  Best β: {beta:.4f}")
    print(f"  Train NLL: {train_nll:.2f}")
    print(f"  Train NLL/event: {train_nll / N_train:.4f}")

    # 3. Compute metrics
    print("\n[3/5] Computing metrics...")
    branching_ratio = compute_branching_ratio(alpha)
    print(f"  Branching ratio (spectral radius of α): {branching_ratio:.6f}")
    print(f"  Stationary: {'Yes' if branching_ratio < 1 else 'No (>1)'}")

    # Held-out log-likelihood
    test_ll = eval_log_likelihood(
        mu, alpha, beta,
        train_times, train_dims,
        test_times, test_dims,
        T_all, D,
    )
    print(f"  Held-out avg log-likelihood: {test_ll:.4f}")

    # Baseline intensities summary
    print(f"\n  Baseline intensities (μ):")
    print(f"    Mean: {mu.mean():.6f}")
    print(f"    Max:  {mu.max():.6f} ({dim_labels[np.argmax(mu)]})")
    print(f"    Min:  {mu.min():.6f}")

    # Alpha sparsity
    n_nonzero = np.sum(alpha > 1e-6)
    print(f"\n  Alpha matrix sparsity:")
    print(f"    Non-zero entries: {n_nonzero}/{D * D} ({100 * n_nonzero / (D * D):.1f}%)")
    print(f"    Max α_ij: {alpha.max():.6f}")
    print(f"    Mean α_ij (non-zero): {alpha[alpha > 1e-6].mean():.6f}" if n_nonzero > 0 else "    All zero")

    # 4. Generate plots
    print("\n[4/5] Generating plots...")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_influence_matrix(alpha, dim_labels, out_dir / "influence_matrix.png", beta)
    plot_baseline_intensities(mu, dim_labels, out_dir / "baseline_intensities.png")
    plot_top_influences(alpha, dim_labels, out_dir / "top_influences.png")

    # 5. Save results
    print("\n[5/5] Saving results...")

    results = {
        "event_id": event_id,
        "event_title": event_title,
        "model": "multivariate_exponential_hawkes",
        "time_axis": "seq (monotonic integer ordering)",
        "time_axis_note": "Median inter-event gap is 0.0s due to second-resolution timestamps; seq provides monotonic ordering",
        "dimensions": D,
        "dimension_type": "(wallet, suit) pairs",
        "n_top_wallets": args.top_wallets,
        "min_events_per_pair": args.min_events,
        "n_train": N_train,
        "n_test": N_test,
        "T_train": float(T_train),
        "fitted_parameters": {
            "beta": float(beta),
            "beta_grid_searched": beta_grid,
            "mu": mu.tolist(),
            "branching_ratio": float(branching_ratio),
            "alpha_nonzero_count": int(n_nonzero),
            "alpha_nonzero_pct": float(100 * n_nonzero / (D * D)),
        },
        "metrics": {
            "train_nll": float(train_nll),
            "train_nll_per_event": float(train_nll / N_train),
            "held_out_avg_log_likelihood": float(test_ll),
        },
        "dimension_labels": [
            {"idx": i, "wallet": w, "suit": s}
            for i, (w, s) in enumerate(dim_labels)
        ],
        "l1_penalty": args.l1,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Save alpha matrix as numpy
    alpha_path = out_dir / "alpha_matrix.npy"
    np.save(alpha_path, alpha)
    print(f"  Saved: {alpha_path}")

    mu_path = out_dir / "mu_vector.npy"
    np.save(mu_path, mu)
    print(f"  Saved: {mu_path}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Dimensions:                  {D}")
    print(f"  Beta (decay):                {beta:.4f}")
    print(f"  Branching ratio:             {branching_ratio:.6f}")
    print(f"  Held-out avg log-likelihood: {test_ll:.4f}")
    print(f"  Alpha sparsity:              {100 * n_nonzero / (D * D):.1f}% non-zero")
    print(f"  Reports:                     {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
