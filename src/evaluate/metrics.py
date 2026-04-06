"""Evaluation metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass


def brier_score(forecast_prob: float, actual_outcome: float) -> float:
    """Compute Brier score between a probabilistic forecast and binary outcome.

    Args:
        forecast_prob: Predicted probability (0-1).
        actual_outcome: Actual binary outcome (0 or 1).

    Returns:
        Brier score (lower is better, 0 = perfect).
    """
    return (forecast_prob - actual_outcome) ** 2


@dataclass
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int


def calibration_curve(
    forecasts: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute calibration curve: expected vs observed frequencies per bin.

    Args:
        forecasts: List of predicted probabilities (0-1).
        outcomes: List of actual binary outcomes (0 or 1).
        n_bins: Number of equal-width bins.

    Returns:
        List of CalibrationBin with expected vs observed frequencies.
    """
    if len(forecasts) != len(outcomes):
        raise ValueError("forecasts and outcomes must have the same length")

    bins: list[CalibrationBin] = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width

        indices = [
            j
            for j, f in enumerate(forecasts)
            if (lower <= f < upper) or (i == n_bins - 1 and f == upper)
        ]

        if not indices:
            bins.append(CalibrationBin(lower, upper, 0.0, 0.0, 0))
            continue

        mean_pred = sum(forecasts[j] for j in indices) / len(indices)
        mean_obs = sum(outcomes[j] for j in indices) / len(indices)
        bins.append(CalibrationBin(lower, upper, mean_pred, mean_obs, len(indices)))

    return bins


@dataclass
class ProfitResult:
    total_pnl: float
    win_rate: float
    roi: float
    n_bets: int
    n_wins: int


def profit_simulation(
    forecasts: list[float],
    outcomes: list[float],
    market_prices: list[float],
    bet_size: float = 100.0,
    divergence_threshold: float = 0.10,
) -> ProfitResult:
    """Simulate profit from betting where forecast diverges from market price.

    Strategy: when |forecast - market_price| > threshold, bet that the forecast
    is correct. Buy YES if forecast > price, buy NO if forecast < price.

    Args:
        forecasts: Model predicted probabilities (0-1).
        outcomes: Actual binary outcomes (0 or 1).
        market_prices: Market prices at time of bet (0-1).
        bet_size: Dollars wagered per qualifying bet.
        divergence_threshold: Minimum |forecast - price| to trigger a bet.

    Returns:
        ProfitResult with total P&L, win rate, ROI.
    """
    if not (len(forecasts) == len(outcomes) == len(market_prices)):
        raise ValueError("forecasts, outcomes, and market_prices must have the same length")

    total_pnl = 0.0
    n_bets = 0
    n_wins = 0

    for forecast, outcome, price in zip(forecasts, outcomes, market_prices):
        divergence = forecast - price
        if abs(divergence) < divergence_threshold:
            continue

        n_bets += 1

        if divergence > 0:
            # Forecast says YES is underpriced — buy YES at `price`
            # Payout: bet_size / price if outcome=1, else lose bet_size
            if outcome == 1:
                payout = bet_size * (1.0 / price - 1.0)
                n_wins += 1
            else:
                payout = -bet_size
        else:
            # Forecast says NO is underpriced — buy NO at `(1 - price)`
            # Payout: bet_size / (1 - price) if outcome=0, else lose bet_size
            if outcome == 0:
                payout = bet_size * (1.0 / (1.0 - price) - 1.0)
                n_wins += 1
            else:
                payout = -bet_size

        total_pnl += payout

    total_wagered = n_bets * bet_size
    win_rate = n_wins / n_bets if n_bets > 0 else 0.0
    roi = total_pnl / total_wagered if total_wagered > 0 else 0.0

    return ProfitResult(
        total_pnl=total_pnl,
        win_rate=win_rate,
        roi=roi,
        n_bets=n_bets,
        n_wins=n_wins,
    )


def mean_brier_score(scores: list[float]) -> float:
    """Average Brier score across markets."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def calibration_error(bins: list[CalibrationBin]) -> float:
    """Weighted mean absolute calibration error across bins."""
    total_count = sum(b.count for b in bins)
    if total_count == 0:
        return 0.0
    return sum(
        b.count * abs(b.mean_predicted - b.mean_observed) for b in bins
    ) / total_count


def sharpe_like_ratio(pnl_per_bet: list[float]) -> float:
    """Sharpe-like ratio: mean return / std of returns."""
    if len(pnl_per_bet) < 2:
        return 0.0
    mean = sum(pnl_per_bet) / len(pnl_per_bet)
    variance = sum((x - mean) ** 2 for x in pnl_per_bet) / (len(pnl_per_bet) - 1)
    std = variance**0.5
    if std == 0:
        return 0.0
    return mean / std
