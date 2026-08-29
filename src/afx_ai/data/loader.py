"""Data loading interface.

Implement `DataLoader.load(ticker)` against a real vendor/broker feed to go
live. `SyntheticDataLoader` is provided for development, testing, and CI so
the full pipeline is runnable without external dependencies or API keys.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from afx_ai.data.synthetic import generate_ohlcv


class DataLoader(ABC):
    """Interface every data source must implement."""

    @abstractmethod
    def load(self, ticker: str) -> pd.DataFrame:
        """Return a DataFrame indexed by date with columns:
        open, high, low, close, volume, ticker."""
        raise NotImplementedError

    def load_many(self, tickers: List[str]) -> dict[str, pd.DataFrame]:
        return {t: self.load(t) for t in tickers}


class SyntheticDataLoader(DataLoader):
    """Deterministic synthetic data — safe default, no network access needed."""

    def __init__(self, n_days: int = 750):
        self.n_days = n_days

    def load(self, ticker: str) -> pd.DataFrame:
        return generate_ohlcv(ticker, n_days=self.n_days)


class CSVDataLoader(DataLoader):
    """Load OHLCV data from local CSV files, one per ticker.

    Expects columns: date, open, high, low, close, volume.
    """

    def __init__(self, directory: str):
        self.directory = directory

    def load(self, ticker: str) -> pd.DataFrame:
        import os

        path = os.path.join(self.directory, f"{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        df["ticker"] = ticker
        return df
