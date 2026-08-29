"""Backtest performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    n_periods = len(equity_curve)
    if n_periods < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    return float(total_return ** (1 / years) - 1) if years > 0 else 0.0


def hit_rate(predicted_up: np.ndarray, actual_up: np.ndarray) -> float:
    return float((predicted_up == actual_up).mean())


def summarize(returns: pd.Series, equity_curve: pd.Series) -> dict:
    return {
        "sharpe_ratio": round(sharpe_ratio(returns), 3),
        "cagr": round(cagr(equity_curve) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(equity_curve) * 100, 2),
        "total_return_pct": round(float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100, 2),
    }
