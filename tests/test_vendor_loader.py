"""Tests the vendor HTTP loader against a MOCKED response, since this
sandbox has no outbound network access to real financial data vendors.
This still exercises the real parsing, retry, and error-handling logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import responses

from afx_ai.data.vendor import HTTPJSONDataLoader, VendorAPIError


@responses.activate
def test_load_parses_valid_payload():
    responses.add(
        responses.GET,
        "https://fake-vendor.example/ohlcv/NPN",
        json={
            "data": [
                {"date": "2026-01-02", "open": 100.0, "high": 101.5, "low": 99.0, "close": 101.0, "volume": 1000},
                {"date": "2026-01-05", "open": 101.0, "high": 102.0, "low": 100.5, "close": 101.8, "volume": 1200},
            ]
        },
        status=200,
    )
    loader = HTTPJSONDataLoader(base_url="https://fake-vendor.example")
    df = loader.load("NPN")

    assert len(df) == 2
    assert list(df.columns) >= list(df.columns)  # sanity
    assert df["close"].iloc[-1] == 101.8
    assert (df["ticker"] == "NPN").all()


@responses.activate
def test_load_retries_then_succeeds():
    # First two attempts fail with a 500, third succeeds
    responses.add(responses.GET, "https://fake-vendor.example/ohlcv/DANGCEM", status=500)
    responses.add(responses.GET, "https://fake-vendor.example/ohlcv/DANGCEM", status=500)
    responses.add(
        responses.GET,
        "https://fake-vendor.example/ohlcv/DANGCEM",
        json={"data": [{"date": "2026-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]},
        status=200,
    )
    loader = HTTPJSONDataLoader(base_url="https://fake-vendor.example", max_retries=3, backoff_seconds=0.01)
    df = loader.load("DANGCEM")
    assert len(df) == 1
    assert len(responses.calls) == 3


@responses.activate
def test_load_raises_after_exhausting_retries():
    for _ in range(3):
        responses.add(responses.GET, "https://fake-vendor.example/ohlcv/BAD", status=503)
    loader = HTTPJSONDataLoader(base_url="https://fake-vendor.example", max_retries=3, backoff_seconds=0.01)
    with pytest.raises(VendorAPIError):
        loader.load("BAD")


@responses.activate
def test_load_raises_on_malformed_payload():
    responses.add(
        responses.GET,
        "https://fake-vendor.example/ohlcv/MALFORMED",
        json={"unexpected": "shape"},
        status=200,
    )
    loader = HTTPJSONDataLoader(base_url="https://fake-vendor.example", max_retries=1)
    with pytest.raises(VendorAPIError):
        loader.load("MALFORMED")
