"""Generic HTTP/JSON vendor data loader.

Implements the "Phase 1: real data integration" TODO item as a concrete,
working, unit-tested loader against a configurable REST endpoint. Most
market-data vendors (broker APIs, Alpha Vantage-style providers, exchange
bulletin APIs) return OHLCV as JSON keyed by date — this class covers that
shape generically; subclass and override `_build_url` / `_parse_payload`
for a vendor whose response shape differs.

Honesty note: this environment's network egress is restricted to package
registries and GitHub (no financial data vendor domains are reachable from
here), so this loader cannot be smoke-tested against a *live* endpoint from
this sandbox. It is fully unit-tested against a mocked HTTP response
(tests/test_vendor_loader.py) exercising the real parsing/retry logic.
Point `base_url` at your vendor and set `HTTPJSONDataLoader.api_key` (or an
env var, if your vendor requires one) to go live from an environment with
outbound access to that vendor.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import pandas as pd
import requests

from afx_ai.data.loader import DataLoader


class VendorAPIError(RuntimeError):
    pass


class HTTPJSONDataLoader(DataLoader):
    """Generic OHLCV loader for REST/JSON vendor APIs.

    Expects (by default) a response shape like:
        {
          "data": [
            {"date": "2026-01-02", "open": 100.1, "high": 101.0,
             "low": 99.5, "close": 100.8, "volume": 123456},
            ...
          ]
        }

    Pass a custom `url_builder` and/or `payload_parser` to adapt to a
    different vendor response shape without subclassing.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        url_builder: Optional[Callable[[str, str], str]] = None,
        payload_parser: Optional[Callable[[dict], pd.DataFrame]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._url_builder = url_builder or self._default_url_builder
        self._payload_parser = payload_parser or self._default_payload_parser
        self.session = session or requests.Session()

    def _default_url_builder(self, base_url: str, ticker: str) -> str:
        url = f"{base_url}/ohlcv/{ticker}"
        if self.api_key:
            url += f"?apikey={self.api_key}"
        return url

    def _default_payload_parser(self, payload: dict) -> pd.DataFrame:
        rows = payload.get("data")
        if rows is None:
            raise VendorAPIError(f"Unexpected payload shape, missing 'data' key: {list(payload)}")
        df = pd.DataFrame(rows)
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise VendorAPIError(f"Vendor response missing required columns: {missing}")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def load(self, ticker: str) -> pd.DataFrame:
        url = self._url_builder(self.base_url, ticker)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                df = self._payload_parser(response.json())
                df["ticker"] = ticker
                return df
            except (requests.RequestException, VendorAPIError, ValueError) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)  # linear backoff
                continue

        raise VendorAPIError(
            f"Failed to load '{ticker}' from {self.base_url} after {self.max_retries} attempts: {last_error}"
        ) from last_error
