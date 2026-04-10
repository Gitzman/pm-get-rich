"""Export cross-market generalization data for the viz site.

Reads cross-market model results, wallet embeddings, classical summary,
and whale profitability data. Creates Parquet files and copies plots
for the generalization.html page.

Usage:
    uv run python scripts/export_generalization.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.manifold import TSNE


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    viz_data = repo / "viz" / "data"
    viz_data.mkdir(parents=True, exist_ok=True)
    gen_reports = viz_data / "gen_reports"
    gen_reports.mkdir(parents=True, exist_ok=True)

    cross_dir = repo / "data" / "reports" / "generalization" / "cross_market"
    gen_dir = repo / "data" / "reports" / "generalization"
    events_dir = repo / "data" / "events"

    # --- 1. Copy cross-market plots ---
    print("Copying cross-market plots...")
    for src in (cross_dir / "plots").glob("*.png"):
        dst = gen_reports / src.name
        shutil.copy2(src, dst)
        print(f"  {dst.name}")

    # --- 2. Load results ---
    with open(cross_dir / "results.json") as f:
        cross_results = json.load(f)
    with open(gen_dir / "summary.json") as f:
        summary = json.load(f)

    # --- 3. Market comparison Parquet (cross-market LL vs classical per market) ---
    print("\nExporting market comparison...")
    rows = []
    # All 25 classical markets
    for mkt in summary["per_market"]:
        eid = mkt["event_id"]
        classical_ll = mkt["classical"]["held_out_avg_log_likelihood"]
        # Check if this market has neural results (in test set)
        neural_data = cross_results["comparison"]["per_market"].get(eid, {})
        neural_ll = neural_data.get("neural_ll")
        rows.append({
            "event_id": eid,
            "city": mkt["city"],
            "date": mkt["date"],
            "classical_ll": round(classical_ll, 4),
            "neural_ll": round(neural_ll, 4) if neural_ll is not None else None,
            "n_train": mkt["classical"]["n_train"],
            "n_test": mkt["classical"]["n_test"],
            "branching_ratio": round(mkt["classical"]["branching_ratio"], 4),
            "in_test_set": neural_ll is not None,
            "improvement": round(neural_ll - classical_ll, 4) if neural_ll is not None else None,
        })
    market_df = pl.DataFrame(rows)
    market_df.write_parquet(viz_data / "gen_market_comparison.parquet", compression="zstd")
    print(f"  gen_market_comparison.parquet: {len(market_df)} rows")

    # --- 4. Wallet embeddings with profitability ---
    print("\nExporting wallet embeddings with profitability...")
    embeddings = np.load(cross_dir / "embeddings.npy")  # (n_wallets, wallet_dim)
    n_wallets = embeddings.shape[0]

    # Load wallet vocabulary from the data
    all_events = pl.read_parquet(events_dir / "all_events.parquet")
    wallet_counts = (
        all_events.group_by("actor")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 5)
        .sort("n", descending=True)
        .head(500)
    )
    wallet_list = wallet_counts["actor"].to_list()

    # Load whale profitability data
    whale_df = pl.read_parquet(viz_data / "whale_leaderboard.parquet")
    whale_map = {}
    for row in whale_df.iter_rows(named=True):
        whale_map[row["address"]] = row

    # Get per-wallet market counts and event counts
    wallet_market_counts = (
        all_events.group_by("actor")
        .agg([
            pl.len().alias("n_events"),
            pl.col("event_id").n_unique().alias("n_markets"),
        ])
    )
    wmc_map = {
        r["actor"]: (r["n_events"], r["n_markets"])
        for r in wallet_market_counts.iter_rows(named=True)
    }

    # Compute t-SNE on full embeddings (excluding UNK token at end)
    n_embed = min(n_wallets - 1, len(wallet_list))  # exclude UNK
    embeds_for_tsne = embeddings[:n_embed]
    perplexity = min(30, n_embed - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeds_for_tsne)

    embed_rows = []
    for i in range(n_embed):
        addr = wallet_list[i] if i < len(wallet_list) else f"wallet_{i}"
        whale_info = whale_map.get(addr, {})
        wmc = wmc_map.get(addr, (0, 0))
        embed_rows.append({
            "idx": i,
            "address": addr,
            "address_short": addr[:10] + "..." + addr[-4:] if len(addr) > 16 else addr,
            "tsne_x": round(float(coords[i, 0]), 4),
            "tsne_y": round(float(coords[i, 1]), 4),
            "n_events": wmc[0],
            "n_markets": wmc[1],
            "total_pnl": round(whale_info.get("total_pnl", 0.0), 2),
            "roi": round(whale_info.get("roi", 0.0), 4),
            "profitable": whale_info.get("total_pnl", 0.0) > 0,
        })

    embed_df = pl.DataFrame(embed_rows)
    embed_df.write_parquet(viz_data / "gen_wallet_embeddings.parquet", compression="zstd")
    print(f"  gen_wallet_embeddings.parquet: {len(embed_df)} rows")

    # --- 5. Attention influence examples ---
    print("\nExporting attention influence examples...")
    # Load model and run attention extraction on held-out test sequences
    import torch

    device = torch.device("cpu")  # CPU is fine for inference

    # Reconstruct data loading to get test set
    from scripts.fit_cross_market_neural import load_cross_market_data, CrossMarketTPP

    data = load_cross_market_data(events_dir, top_k_wallets=500)
    model = CrossMarketTPP(
        n_wallets=data["n_wallets"],
        n_cities=data["n_cities"],
        d_model=cross_results["architecture"]["d_model"],
        n_heads=cross_results["architecture"]["n_heads"],
        n_layers=cross_results["architecture"]["n_layers"],
        d_ff=cross_results["architecture"]["d_ff"],
        wallet_dim=cross_results["architecture"]["wallet_dim"],
    )
    model.load_state_dict(torch.load(cross_dir / "model.pt", map_location=device, weights_only=True))
    model.eval()

    split = data["split_idx"]
    ctx = 128
    n_examples = 5  # number of attention example windows

    attention_rows = []
    for ex_idx in range(n_examples):
        start = split + ex_idx * ctx
        end = min(start + ctx, data["N"] - 1)
        L = end - start
        if L < 32:
            break

        inp = {
            "wallet_idx": torch.tensor(data["wallet_indices"][start:end][np.newaxis], dtype=torch.long),
            "city_idx": torch.tensor(data["city_indices"][start:end][np.newaxis], dtype=torch.long),
            "side_idx": torch.tensor(data["side_indices"][start:end][np.newaxis], dtype=torch.long),
            "bucket_pos": torch.tensor(data["bucket_positions"][start:end][np.newaxis], dtype=torch.float32),
            "price": torch.tensor(data["prices"][start:end][np.newaxis], dtype=torch.float32),
            "time_delta": torch.tensor(data["time_deltas"][start:end][np.newaxis], dtype=torch.float32),
            "hours_to_res": torch.tensor(data["hours_to_resolution"][start:end][np.newaxis], dtype=torch.float32),
            "n_buckets": torch.tensor(data["n_buckets"][start:end][np.newaxis], dtype=torch.float32),
        }

        with torch.no_grad():
            attn_weights = model.get_attention_weights(**inp)
        # Average across layers and heads: (L, L)
        avg_attn = torch.stack(attn_weights).mean(dim=(0, 1, 2)).cpu().numpy()

        # Find top attention pairs where source wallet != target wallet
        for query_pos in range(L):
            query_wallet_idx = data["wallet_indices"][start + query_pos]
            query_wallet = data["wallet_list"][query_wallet_idx] if query_wallet_idx < len(data["wallet_list"]) else "UNK"
            query_event = data["event_ids"][start + query_pos]
            query_city = data["event_meta"].get(query_event, {}).get("city", "?")

            # Top 3 attended-to positions
            attn_row = avg_attn[query_pos, :query_pos + 1]
            if attn_row.sum() < 1e-8:
                continue
            top_keys = np.argsort(attn_row)[::-1][:5]
            for key_pos in top_keys:
                key_wallet_idx = data["wallet_indices"][start + key_pos]
                if key_wallet_idx == query_wallet_idx:
                    continue  # skip self-attention
                key_wallet = data["wallet_list"][key_wallet_idx] if key_wallet_idx < len(data["wallet_list"]) else "UNK"
                key_event = data["event_ids"][start + key_pos]
                key_city = data["event_meta"].get(key_event, {}).get("city", "?")
                attention_rows.append({
                    "example": ex_idx,
                    "query_wallet": query_wallet[:10] + "..." if len(query_wallet) > 12 else query_wallet,
                    "query_city": query_city,
                    "query_event": query_event,
                    "key_wallet": key_wallet[:10] + "..." if len(key_wallet) > 12 else key_wallet,
                    "key_city": key_city,
                    "key_event": key_event,
                    "attention_weight": round(float(attn_row[key_pos]), 6),
                    "cross_market": query_event != key_event,
                    "cross_city": query_city != key_city,
                    "time_gap_positions": query_pos - key_pos,
                })

    if attention_rows:
        attn_df = pl.DataFrame(attention_rows)
        # Keep top 200 by attention weight
        attn_df = attn_df.sort("attention_weight", descending=True).head(200)
        attn_df.write_parquet(viz_data / "gen_attention_examples.parquet", compression="zstd")
        print(f"  gen_attention_examples.parquet: {len(attn_df)} rows")

    # --- 6. Per-city aggregate comparison ---
    print("\nExporting per-city summary...")
    city_rows = []
    for city, data_city in summary["per_city"].items():
        # Compute neural mean for test markets in this city
        neural_lls = []
        for eid, mdata in cross_results["comparison"]["per_market"].items():
            if mdata["city"] == city:
                neural_lls.append(mdata["neural_ll"])
        city_rows.append({
            "city": city,
            "n_markets": data_city["n_markets"],
            "classical_mean_ll": round(data_city["classical_mean_ll"], 4),
            "neural_mean_ll": round(float(np.mean(neural_lls)), 4) if neural_lls else None,
            "n_neural_test": len(neural_lls),
        })
    city_df = pl.DataFrame(city_rows)
    city_df.write_parquet(viz_data / "gen_city_comparison.parquet", compression="zstd")
    print(f"  gen_city_comparison.parquet: {len(city_df)} rows")

    # --- 7. Model summary ---
    print("\nExporting model summary...")
    model_summary = pl.DataFrame([{
        "metric": "Architecture",
        "value": f"{cross_results['architecture']['n_layers']}-layer Transformer ({cross_results['architecture']['d_model']}d, {cross_results['architecture']['n_heads']} heads)",
    }, {
        "metric": "Parameters",
        "value": f"{cross_results['n_parameters']:,}",
    }, {
        "metric": "Markets trained on",
        "value": str(cross_results["data"]["n_markets"]),
    }, {
        "metric": "Total events",
        "value": f"{cross_results['data']['n_events_total']:,}",
    }, {
        "metric": "Train / Test split",
        "value": f"{cross_results['data']['n_train']:,} / {cross_results['data']['n_test']:,}",
    }, {
        "metric": "Wallet vocabulary",
        "value": f"{cross_results['data']['n_wallets_vocab']} (top 500 + UNK)",
    }, {
        "metric": "Cities",
        "value": f"{cross_results['data']['n_cities']} ({', '.join(cross_results['data']['cities'])})",
    }, {
        "metric": "Cross-market held-out LL",
        "value": f"{cross_results['metrics']['global_held_out_ll']:.4f}",
    }, {
        "metric": "Classical baseline LL (mean)",
        "value": f"{cross_results['comparison']['classical_mean_ll']:.4f}",
    }, {
        "metric": "Neural beats classical",
        "value": f"{sum(1 for m in cross_results['comparison']['per_market'].values() if m['neural_ll'] > m['classical_ll'])}/{len(cross_results['comparison']['per_market'])} test markets",
    }, {
        "metric": "Training time",
        "value": f"{cross_results['training']['training_time_seconds']:.0f}s on {cross_results['gpu']['name']}",
    }, {
        "metric": "Peak VRAM",
        "value": f"{cross_results['gpu']['peak_vram_mb']:.0f} MB",
    }])
    model_summary.write_parquet(viz_data / "gen_model_summary.parquet", compression="zstd")
    print(f"  gen_model_summary.parquet: {len(model_summary)} rows")

    # Summary
    total = sum(f.stat().st_size for f in viz_data.glob("gen_*.parquet"))
    img_total = sum(f.stat().st_size for f in gen_reports.glob("*.png"))
    print(f"\nTotal Parquet: {total / 1024:.0f} KB")
    print(f"Total images: {img_total / 1024:.0f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
