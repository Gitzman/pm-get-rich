"""Report generation for backtest results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluate.backtest import BacktestSummary, MarketResult


def print_top_bottom_markets(results: list[MarketResult], n: int = 10) -> None:
    """Print formatted table of top/bottom markets by Brier score."""
    sorted_results = sorted(results, key=lambda r: r.brier)

    print(f"\n{'='*90}")
    print(f"  TOP {n} MARKETS (lowest Brier score = best forecasts)")
    print(f"{'='*90}")
    _print_market_table(sorted_results[:n])

    print(f"\n{'='*90}")
    print(f"  BOTTOM {n} MARKETS (highest Brier score = worst forecasts)")
    print(f"{'='*90}")
    _print_market_table(sorted_results[-n:])


def _print_market_table(markets: list[MarketResult]) -> None:
    """Print a formatted table of market results."""
    header = f"{'Market':<45} {'Forecast':>8} {'Outcome':>7} {'Price':>6} {'Brier':>7} {'Pts':>5}"
    print(header)
    print("-" * len(header))
    for r in markets:
        question = r.question[:43] + ".." if len(r.question) > 45 else r.question
        print(
            f"{question:<45} {r.forecast_median:>8.3f} {r.outcome:>7.0f} "
            f"{r.last_price:>6.3f} {r.brier:>7.4f} {r.n_datapoints:>5}"
        )


def print_calibration_table(summary: BacktestSummary) -> None:
    """Print calibration curve as ASCII table."""
    print(f"\n{'='*60}")
    print("  CALIBRATION CURVE")
    print(f"{'='*60}")
    header = f"{'Bin':>12} {'Predicted':>10} {'Observed':>10} {'Count':>7} {'Bar'}"
    print(header)
    print("-" * 60)

    for b in summary.calibration_bins:
        bin_label = f"[{b.bin_lower:.1f}-{b.bin_upper:.1f})"
        bar_len = int(b.mean_observed * 40) if b.count > 0 else 0
        bar = "#" * bar_len
        print(
            f"{bin_label:>12} {b.mean_predicted:>10.3f} {b.mean_observed:>10.3f} "
            f"{b.count:>7} {bar}"
        )

    print(f"\n  Calibration Error (weighted MAE): {summary.cal_error:.4f}")


def print_profit_summary(summary: BacktestSummary) -> None:
    """Print profit simulation summary."""
    p = summary.profit
    print(f"\n{'='*60}")
    print("  PROFIT SIMULATION")
    print(f"{'='*60}")
    pnl_sign = "+" if p.total_pnl >= 0 else ""
    print(f"  Strategy: bet $100 on every market where Chronos")
    print(f"            diverged from market price by >threshold")
    print()
    print(f"  Total P&L:     {pnl_sign}${p.total_pnl:,.2f}")
    print(f"  Bets placed:   {p.n_bets}")
    print(f"  Wins:          {p.n_wins}")
    print(f"  Win rate:      {p.win_rate:.1%}")
    print(f"  ROI:           {p.roi:.1%}")
    print(f"  Sharpe-like:   {summary.sharpe:.3f}")


def print_summary_stats(summary: BacktestSummary) -> None:
    """Print overall summary statistics."""
    print(f"\n{'='*60}")
    print("  SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Markets evaluated:     {len(summary.results)}")
    print(f"  Mean Brier score:      {summary.avg_brier:.4f}")
    print(f"  Calibration error:     {summary.cal_error:.4f}")

    p = summary.profit
    pnl_sign = "+" if p.total_pnl >= 0 else ""
    verb = "made" if p.total_pnl >= 0 else "lost"
    print()
    print(f"  >>> If we bet $100 on every market where Chronos diverged")
    print(f"  >>> by >threshold, we would have {verb} ${abs(p.total_pnl):,.2f}")
    print(f"  >>> with {p.win_rate:.1%} win rate")


def print_full_report(summary: BacktestSummary) -> None:
    """Print the complete backtest report."""
    print_top_bottom_markets(summary.results)
    print_calibration_table(summary)
    print_profit_summary(summary)
    print_summary_stats(summary)


def save_html_report(summary: BacktestSummary, output_path: Path) -> None:
    """Save a simple HTML report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(summary.results, key=lambda r: r.brier)
    p = summary.profit
    pnl_sign = "+" if p.total_pnl >= 0 else ""
    verb = "made" if p.total_pnl >= 0 else "lost"

    rows_html = ""
    for r in sorted_results:
        rows_html += (
            f"<tr><td>{r.question}</td><td>{r.forecast_median:.3f}</td>"
            f"<td>{r.outcome:.0f}</td><td>{r.last_price:.3f}</td>"
            f"<td>{r.brier:.4f}</td><td>{r.n_datapoints}</td></tr>\n"
        )

    cal_rows = ""
    for b in summary.calibration_bins:
        cal_rows += (
            f"<tr><td>[{b.bin_lower:.1f}-{b.bin_upper:.1f})</td>"
            f"<td>{b.mean_predicted:.3f}</td><td>{b.mean_observed:.3f}</td>"
            f"<td>{b.count}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><title>Backtest Report</title>
<style>
body {{ font-family: monospace; max-width: 1200px; margin: 0 auto; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
th {{ background: #f0f0f0; }}
td:first-child {{ text-align: left; }}
h2 {{ margin-top: 30px; }}
.summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
</style></head><body>
<h1>Backtest Report</h1>
<div class="summary">
<p><strong>Markets evaluated:</strong> {len(summary.results)}</p>
<p><strong>Mean Brier score:</strong> {summary.avg_brier:.4f}</p>
<p><strong>Calibration error:</strong> {summary.cal_error:.4f}</p>
<p><strong>Result:</strong> If we bet $100 on every divergent market,
we would have {verb} {pnl_sign}${abs(p.total_pnl):,.2f} with {p.win_rate:.1%} win rate
(ROI: {p.roi:.1%}, Sharpe-like: {summary.sharpe:.3f})</p>
</div>

<h2>All Markets (sorted by Brier score)</h2>
<table>
<tr><th>Question</th><th>Forecast</th><th>Outcome</th><th>Price</th><th>Brier</th><th>Points</th></tr>
{rows_html}</table>

<h2>Calibration Curve</h2>
<table>
<tr><th>Bin</th><th>Predicted</th><th>Observed</th><th>Count</th></tr>
{cal_rows}</table>

<h2>Profit Simulation</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total P&amp;L</td><td>{pnl_sign}${p.total_pnl:,.2f}</td></tr>
<tr><td>Bets placed</td><td>{p.n_bets}</td></tr>
<tr><td>Win rate</td><td>{p.win_rate:.1%}</td></tr>
<tr><td>ROI</td><td>{p.roi:.1%}</td></tr>
<tr><td>Sharpe-like</td><td>{summary.sharpe:.3f}</td></tr>
</table>
</body></html>"""

    output_path.write_text(html)
    print(f"\n  HTML report saved to {output_path}")
