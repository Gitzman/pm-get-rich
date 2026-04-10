"""Export Hawkes comparison data as Parquet for the viz site.

Reads classical and neural results, creates queryable Parquet files
and copies report images for the comparison page.

Usage:
    uv run python scripts/export_comparison.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    viz_data = repo / "viz" / "data"
    viz_data.mkdir(parents=True, exist_ok=True)
    reports_dir = viz_data / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    classical_dir = repo / "data" / "reports" / "hawkes_classical"
    neural_dir = repo / "data" / "reports" / "hawkes_neural"
    events_path = repo / "data" / "events" / "295980" / "events.parquet"

    # --- 1. Copy report images ---
    print("Copying report images...")
    for src in classical_dir.glob("*.png"):
        dst = reports_dir / f"classical_{src.name}"
        shutil.copy2(src, dst)
        print(f"  {dst.name}")
    for src in neural_dir.glob("*.png"):
        dst = reports_dir / f"neural_{src.name}"
        shutil.copy2(src, dst)
        print(f"  {dst.name}")

    # --- 2. Load results ---
    with open(classical_dir / "results.json") as f:
        classical = json.load(f)
    with open(neural_dir / "results.json") as f:
        neural = json.load(f)

    # --- 3. Metrics comparison Parquet ---
    print("\nExporting comparison metrics...")
    metrics = pl.DataFrame({
        "metric": [
            "Held-out Avg Log-Likelihood",
            "Train NLL per Event",
            "Dimensions",
            "Training Events",
            "Test Events",
            "Branching Ratio",
            "Training Time (s)",
            "Parameters",
            "Model",
        ],
        "classical": [
            str(round(classical["metrics"]["held_out_avg_log_likelihood"], 4)),
            str(round(classical["metrics"]["train_nll_per_event"], 4)),
            str(classical["dimensions"]),
            str(classical["n_train"]),
            str(classical["n_test"]),
            str(round(classical["fitted_parameters"]["branching_ratio"], 4)),
            "~120",  # grid search across 7 betas
            str(classical["dimensions"] + classical["dimensions"] ** 2 + 1),  # mu + alpha + beta
            classical["model"],
        ],
        "neural": [
            str(round(neural["metrics"]["held_out_avg_log_likelihood"], 4)),
            str(round(neural["training"]["final_train_nll_per_event"], 4)),
            str(neural["dimensions"]),
            str(neural["n_train"]),
            str(neural["n_test"]),
            "N/A (implicit)",
            str(round(neural["training"]["training_time_seconds"])),
            str(neural["n_parameters"]),
            neural["model"],
        ],
    })
    metrics.write_parquet(viz_data / "hawkes_metrics.parquet", compression="zstd")
    print(f"  hawkes_metrics.parquet: {len(metrics)} rows")

    # --- 4. Classical top influences Parquet ---
    print("Exporting classical top influences...")
    alpha = np.load(classical_dir / "alpha_matrix.npy")
    dim_labels = classical["dimension_labels"]
    D = alpha.shape[0]

    # Top 50 influence pairs
    flat_idx = np.argsort(alpha.ravel())[::-1][:50]
    rows_idx, cols_idx = np.unravel_index(flat_idx, (D, D))

    influence_rows = []
    for r, c in zip(rows_idx, cols_idx):
        val = float(alpha[r, c])
        if val < 1e-6:
            break
        src = dim_labels[int(c)]
        tgt = dim_labels[int(r)]
        influence_rows.append({
            "rank": len(influence_rows) + 1,
            "source_wallet": src["wallet"],
            "source_suit": src["suit"],
            "target_wallet": tgt["wallet"],
            "target_suit": tgt["suit"],
            "alpha": round(val, 6),
            "source_label": f"{src['wallet']}_{src['suit']}",
            "target_label": f"{tgt['wallet']}_{tgt['suit']}",
        })

    influences_df = pl.DataFrame(influence_rows)
    influences_df.write_parquet(viz_data / "classical_influences.parquet", compression="zstd")
    print(f"  classical_influences.parquet: {len(influences_df)} rows")

    # --- 5. Event snippets for top-K pairs ---
    print("Exporting event snippets for top influence pairs...")
    events = pl.read_parquet(events_path)

    # Get unique (source_wallet, target_wallet) from top influences
    top_wallets = set()
    for row in influence_rows[:20]:
        top_wallets.add(row["source_wallet"])
        top_wallets.add(row["target_wallet"])

    snippets = (
        events
        .filter(pl.col("actor").str.starts_with(w) for w in list(top_wallets)[:1])
    )
    # Simpler: filter by any of the top wallets (partial match on short addr)
    wallet_list = list(top_wallets)
    mask = pl.lit(False)
    for w in wallet_list:
        mask = mask | pl.col("actor").str.contains(w, literal=True)

    snippets = events.filter(mask).sort("seq").head(500)
    snippets.write_parquet(viz_data / "top_pair_events.parquet", compression="zstd")
    print(f"  top_pair_events.parquet: {len(snippets)} rows")

    # --- 6. Dimension labels with baseline intensities ---
    print("Exporting dimension data...")
    mu = np.load(classical_dir / "mu_vector.npy")
    dim_data = []
    for i, label in enumerate(dim_labels):
        dim_data.append({
            "idx": label["idx"],
            "wallet": label["wallet"],
            "suit": label["suit"],
            "classical_mu": round(float(mu[i]), 8),
        })
    dim_df = pl.DataFrame(dim_data)
    dim_df.write_parquet(viz_data / "hawkes_dimensions.parquet", compression="zstd")
    print(f"  hawkes_dimensions.parquet: {len(dim_df)} rows")

    # Summary
    total = sum(f.stat().st_size for f in viz_data.glob("*.parquet"))
    img_total = sum(f.stat().st_size for f in reports_dir.glob("*.png"))
    print(f"\nTotal Parquet: {total / 1024:.0f} KB")
    print(f"Total images: {img_total / 1024:.0f} KB")
    print(f"Files in {viz_data}/")


if __name__ == "__main__":
    main()
