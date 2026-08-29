import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.data.loader import DataLoader
from afx_ai.data.cache import CachedDataLoader
from afx_ai.data.synthetic import generate_ohlcv


class _CountingLoader(DataLoader):
    """Wraps the synthetic generator but counts real (non-cached) calls."""
    def __init__(self):
        self.calls = 0

    def load(self, ticker: str):
        self.calls += 1
        return generate_ohlcv(ticker, n_days=50, seed=1)


def test_cache_avoids_refetch_within_ttl(tmp_path):
    inner = _CountingLoader()
    loader = CachedDataLoader(inner, cache_dir=str(tmp_path), ttl_seconds=3600)

    df1 = loader.load("NPN")
    df2 = loader.load("NPN")

    assert inner.calls == 1  # second load served from cache
    assert len(df1) == len(df2)


def test_cache_refetches_after_ttl_expires(tmp_path):
    inner = _CountingLoader()
    loader = CachedDataLoader(inner, cache_dir=str(tmp_path), ttl_seconds=0.05)

    loader.load("NPN")
    time.sleep(0.1)
    loader.load("NPN")

    assert inner.calls == 2


def test_invalidate_forces_refetch(tmp_path):
    inner = _CountingLoader()
    loader = CachedDataLoader(inner, cache_dir=str(tmp_path), ttl_seconds=3600)

    loader.load("NPN")
    loader.invalidate("NPN")
    loader.load("NPN")

    assert inner.calls == 2
