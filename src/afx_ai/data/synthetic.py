"""Synthetic OHLCV data generator.

Used so the full pipeline (features -> models -> ensemble -> backtest) runs
end-to-end without requiring a live market data subscription. Replace with a
real feed via the DataLoader interface in loader.py for production use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    ticker: str,
    n_days: int = 750,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.28,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a plausible daily OHLCV series via geometric Brownian motion
    plus mild autocorrelated regime noise, so technical indicators behave
    realistically (trends, pullbacks, volatility clustering)."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**32))

    dt = 1 / 252
    mu, sigma = annual_drift, annual_vol

    # Regime-switching volatility (simple 2-state Markov) for clustering
    regimes = np.zeros(n_days, dtype=int)
    p_switch = 0.02
    for i in range(1, n_days):
        regimes[i] = 1 - regimes[i - 1] if rng.random() < p_switch else regimes[i - 1]
    vol_multiplier = np.where(regimes == 1, 1.8, 0.8)

    shocks = rng.normal(0, 1, n_days)
    daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * vol_multiplier * np.sqrt(dt) * shocks

    close = start_price * np.exp(np.cumsum(daily_returns))
    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.002, n_days - 1))

    intraday_range = np.abs(rng.normal(0, sigma * vol_multiplier * np.sqrt(dt), n_days))
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)
    volume = rng.lognormal(mean=13, sigma=0.5, size=n_days).astype(int)

    # Note: anchoring with `end=` can occasionally be off-by-one when the
    # anchor date isn't a business day itself, so we over-generate from a
    # safely early start date and take the last n_days entries.
    all_dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=n_days + 5
    )
    dates = all_dates[-n_days:]

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ).set_index("date")
    df["ticker"] = ticker
    return df
