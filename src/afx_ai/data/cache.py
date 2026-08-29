"""Local disk cache wrapper for any DataLoader.

Avoids re-fetching unchanged history from a real vendor feed on every daily
run — a real concern once Phase 1's live loader is wired in against a rate-
limited or paid API. Referenced from TODO.md Phase 1.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from afx_ai.data.loader import DataLoader


class CachedDataLoader(DataLoader):
    """Wraps another DataLoader, caching each ticker's result to a local CSV
    keyed by ticker, refreshing only after `ttl_seconds` has elapsed.

    Usage:
        real_loader = SomeVendorLoader(api_key=...)
        loader = CachedDataLoader(real_loader, cache_dir="data/cache", ttl_seconds=6*3600)
        df = loader.load("NPN")   # hits the vendor once, then serves from disk
    """

    def __init__(self, inner: DataLoader, cache_dir: str = "data/cache", ttl_seconds: int = 6 * 3600):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.csv"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < self.ttl_seconds

    def load(self, ticker: str) -> pd.DataFrame:
        path = self._cache_path(ticker)
        if self._is_fresh(path):
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            df["ticker"] = ticker
            return df

        df = self.inner.load(ticker)
        df.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)
        return df

    def invalidate(self, ticker: str | None = None) -> None:
        """Force the next load() to bypass the cache — call after you know
        the underlying data changed (e.g. after a detected corporate action)."""
        if ticker is None:
            for f in self.cache_dir.glob("*.csv"):
                f.unlink()
        else:
            path = self._cache_path(ticker)
            if path.exists():
                path.unlink()
