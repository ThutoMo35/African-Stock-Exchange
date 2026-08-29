"""Vectorized long/flat backtester driven by ensemble signal probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from afx_ai.backtest.metrics import summarize


def run_backtest(
    dates: pd.DatetimeIndex,
    forward_returns: np.ndarray,
    signal_proba_up: np.ndarray,
    threshold: float = 0.55,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Go long when signal_proba_up > threshold, otherwise stay flat.

    forward_returns[i] is the realized next-period return corresponding to
    the decision made using signal_proba_up[i].
    """
    position = (signal_proba_up > threshold).astype(float)
    position_change = np.abs(np.diff(position, prepend=0))
    cost = position_change * (transaction_cost_bps / 10_000)

    strategy_returns = position * forward_returns - cost
    equity_curve = pd.Series((1 + strategy_returns).cumprod(), index=dates)
    returns_series = pd.Series(strategy_returns, index=dates)

    buy_hold_curve = pd.Series((1 + forward_returns).cumprod(), index=dates)

    return {
        "equity_curve": equity_curve,
        "buy_hold_curve": buy_hold_curve,
        "metrics": summarize(returns_series, equity_curve),
        "buy_hold_metrics": summarize(pd.Series(forward_returns, index=dates), buy_hold_curve),
        "avg_exposure": float(position.mean()),
        "num_trades": int(position_change.sum()),
    }
