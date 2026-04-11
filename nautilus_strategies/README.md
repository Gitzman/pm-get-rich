# NautilusTrader Integration for TPP Model

MIT-licensed strategy code for running our CrossMarketTPP model inside the
[prediction-market-backtesting](https://github.com/evan-kolberg/prediction-market-backtesting)
NautilusTrader framework.

## Setup

```bash
# Clone the prediction-market-backtesting framework
git clone https://github.com/evan-kolberg/prediction-market-backtesting.git

# Install dependencies
cd prediction-market-backtesting
uv venv --python 3.12 && uv pip install \
  "nautilus_trader[polymarket]==1.225.0" \
  bokeh plotly numpy onnxruntime pyarrow polars duckdb

# Copy our strategies into the framework
cp ../nautilus_strategies/tpp_signal.py strategies/
cp ../nautilus_strategies/volume_baseline.py strategies/
cp ../nautilus_strategies/polymarket_weather_tpp_vs_volume.py backtests/

# Export ONNX model (if not already done)
cd .. && uv run python scripts/export_tpp_onnx.py

# Run backtest
cd prediction-market-backtesting
.venv/bin/python backtests/polymarket_weather_tpp_vs_volume.py
```

## Files

- `tpp_signal.py` — NautilusTrader Strategy wrapping TPP ONNX model (BUY-only)
- `volume_baseline.py` — Volume spike baseline strategy for comparison
- `polymarket_weather_tpp_vs_volume.py` — Backtest runner for weather markets
- `RESULTS.md` — Full results and analysis report

## Key Findings

See `RESULTS.md` for the complete analysis. Summary: the TPP model suffers severe
feature degradation when deployed through NautilusTrader's standard tick data
abstraction (wallet identity and bucket position features are unavailable),
confirming and strengthening the "no exploitable edge" assessment from our
statistical backtest.
