"""Technical + statistical feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given raw OHLCV, produce a feature matrix. Returns a new DataFrame
    aligned on the same index, with NaNs from warmup periods dropped."""
    out = df.copy()
    close = out["close"]

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)

    out["sma_10"] = close.rolling(10).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_50"]

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["volatility_30"] = out["return_1d"].rolling(30).std()

    out["rsi_14"] = _rsi(close, 14)

    macd, macd_signal, macd_hist = _macd(close)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    out["volume_change"] = out["volume"].pct_change(1)
    out["volume_zscore"] = (
        out["volume"] - out["volume"].rolling(20).mean()
    ) / out["volume"].rolling(20).std()

    out["momentum_10"] = close / close.shift(10) - 1
    out["price_vs_sma50"] = close / out["sma_50"] - 1

    # Target: next-day direction (1 = up, 0 = down) — used by classifiers
    out["target_direction"] = (close.shift(-1) > close).astype(int)
    # Target: next-day return — used by regressors
    out["target_return"] = close.shift(-1) / close - 1

    out = out.dropna()
    return out


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d",
    "sma_ratio", "volatility_10", "volatility_30",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct_b", "volume_change", "volume_zscore",
    "momentum_10", "price_vs_sma50",
]
