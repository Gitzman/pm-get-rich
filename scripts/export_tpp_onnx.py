"""Export CrossMarketTPP model to ONNX for Rust inference via ort crate.

Usage:
    uv run python scripts/export_tpp_onnx.py

Reads:
    data/reports/fullscale/pre_march25/model.pt   (state dict)
    data/reports/generalization/cross_market/vocab.json  (n_wallets, n_cities)

Writes:
    data/models/tpp.onnx
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

MODEL_PT = ROOT / "data" / "reports" / "fullscale" / "pre_march25" / "model.pt"
VOCAB_JSON = ROOT / "data" / "reports" / "generalization" / "cross_market" / "vocab.json"
OUT_DIR = ROOT / "data" / "models"
OUT_ONNX = OUT_DIR / "tpp.onnx"


# ---------------------------------------------------------------------------
# sinusoidal_time_encoding (must match training code exactly)
# ---------------------------------------------------------------------------

def sinusoidal_time_encoding(dt: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of time deltas. Input (*, ), output (*, dim)."""
    log_dt = torch.log1p(dt).unsqueeze(-1)
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=dt.device, dtype=dt.dtype)
        * (-math.log(10000.0) / dim)
    )
    args = log_dt * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Model (identical to fit_cross_market_neural.py)
# ---------------------------------------------------------------------------

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

        self.wallet_head = nn.Linear(d_model, n_wallets)
        self.bucket_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def _encode_events(
        self,
        wallet_idx: torch.Tensor,
        city_idx: torch.Tensor,
        side_idx: torch.Tensor,
        bucket_pos: torch.Tensor,
        price: torch.Tensor,
        time_delta: torch.Tensor,
        hours_to_res: torch.Tensor,
        n_buckets: torch.Tensor,
    ) -> torch.Tensor:
        w_emb = self.wallet_embed(wallet_idx)
        c_emb = self.city_embed(city_idx)
        s_emb = self.side_embed(side_idx)

        b_enc = self.bucket_proj(bucket_pos.unsqueeze(-1))
        p_enc = self.price_proj(price.unsqueeze(-1))

        ctx = torch.stack([
            hours_to_res / 200.0,
            n_buckets / 11.0,
        ], dim=-1)
        ctx_enc = self.context_proj(ctx)

        t_enc = sinusoidal_time_encoding(time_delta, self.time_dim)

        features = torch.cat([w_emb, c_emb, s_emb, b_enc, p_enc, ctx_enc, t_enc], dim=-1)
        return self.input_proj(features)

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
        B, L = wallet_idx.shape

        x = self._encode_events(
            wallet_idx, city_idx, side_idx, bucket_pos, price,
            time_delta, hours_to_res, n_buckets,
        )

        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.transformer(x, mask=mask, is_causal=True)

        wallet_logits = self.wallet_head(h)
        bucket_pred = self.bucket_head(h).squeeze(-1)
        time_params = self.time_head(h)
        time_mu = time_params[..., 0]
        time_log_sigma = time_params[..., 1]

        return wallet_logits, bucket_pred, time_mu, time_log_sigma


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def main() -> None:
    # Load vocab
    with open(VOCAB_JSON) as f:
        vocab = json.load(f)
    n_wallets = vocab["n_wallets"]
    n_cities = vocab["n_cities"]
    print(f"Vocab: {n_wallets} wallets, {n_cities} cities")

    # Build model and load weights
    model = CrossMarketTPP(n_wallets=n_wallets, n_cities=n_cities)
    state_dict = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy inputs (batch=1, seq_len=64 — representative inference size)
    B, L = 1, 64
    dummy_inputs = (
        torch.randint(0, n_wallets, (B, L)),    # wallet_idx
        torch.randint(0, n_cities, (B, L)),      # city_idx
        torch.randint(0, 2, (B, L)),             # side_idx
        torch.rand(B, L),                        # bucket_pos
        torch.rand(B, L),                        # price
        torch.rand(B, L) * 10.0,                 # time_delta
        torch.rand(B, L) * 200.0,               # hours_to_res
        torch.rand(B, L) * 11.0,                # n_buckets
    )
    input_names = [
        "wallet_idx", "city_idx", "side_idx", "bucket_pos",
        "price", "time_delta", "hours_to_res", "n_buckets",
    ]
    output_names = ["wallet_logits", "bucket_pred", "time_mu", "time_log_sigma"]

    # Verify forward pass works
    with torch.no_grad():
        out = model(*dummy_inputs)
    print(f"Forward pass OK — wallet_logits: {out[0].shape}, bucket_pred: {out[1].shape}, "
          f"time_mu: {out[2].shape}, time_log_sigma: {out[3].shape}")

    # Export to ONNX
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamic axes: batch and sequence length can vary
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch", 1: "seq_len"}
    for name in output_names:
        if name == "wallet_logits":
            dynamic_axes[name] = {0: "batch", 1: "seq_len", 2: "n_wallets"}
        else:
            dynamic_axes[name] = {0: "batch", 1: "seq_len"}

    print(f"Exporting to {OUT_ONNX} ...")
    torch.onnx.export(
        model,
        dummy_inputs,
        str(OUT_ONNX),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    print(f"ONNX export complete: {OUT_ONNX} ({OUT_ONNX.stat().st_size / 1024:.1f} KB)")

    # Verify with onnxruntime
    import onnxruntime as ort

    session = ort.InferenceSession(str(OUT_ONNX))
    ort_inputs = {
        name: inp.numpy() for name, inp in zip(input_names, dummy_inputs)
    }
    ort_outputs = session.run(None, ort_inputs)

    # Compare PyTorch vs ONNX outputs
    torch_outputs = [o.numpy() for o in out]
    for i, (name, torch_out, ort_out) in enumerate(
        zip(output_names, torch_outputs, ort_outputs)
    ):
        max_diff = np.max(np.abs(torch_out - ort_out))
        print(f"  {name}: max_diff={max_diff:.2e}  shape={ort_out.shape}")
        assert max_diff < 1e-4, f"ONNX verification failed for {name}: max_diff={max_diff}"

    # Test with different sequence lengths to verify dynamic axes
    for test_L in [16, 128, 512]:
        test_inputs = {
            "wallet_idx": np.random.randint(0, n_wallets, (1, test_L)).astype(np.int64),
            "city_idx": np.random.randint(0, n_cities, (1, test_L)).astype(np.int64),
            "side_idx": np.random.randint(0, 2, (1, test_L)).astype(np.int64),
            "bucket_pos": np.random.rand(1, test_L).astype(np.float32),
            "price": np.random.rand(1, test_L).astype(np.float32),
            "time_delta": (np.random.rand(1, test_L) * 10).astype(np.float32),
            "hours_to_res": (np.random.rand(1, test_L) * 200).astype(np.float32),
            "n_buckets": (np.random.rand(1, test_L) * 11).astype(np.float32),
        }
        test_out = session.run(None, test_inputs)
        assert test_out[0].shape == (1, test_L, n_wallets), \
            f"Dynamic shape test failed for L={test_L}: got {test_out[0].shape}"
        print(f"  Dynamic shape test L={test_L}: OK")

    print("\nAll verification passed. ONNX model is ready for Rust ort crate.")


if __name__ == "__main__":
    main()
