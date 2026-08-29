import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from afx_ai.data.synthetic import generate_ohlcv
from afx_ai.data.quality import (
    run_quality_checks,
    check_stale_prices,
    detect_suspected_splits,
    adjust_for_suspected_splits,
)


def test_clean_synthetic_data_has_no_stale_runs_or_bad_prices():
    df = generate_ohlcv("CLEAN", n_days=300, seed=42)
    report = run_quality_checks(df, "CLEAN")
    assert report.negative_or_zero_prices == []
    assert report.high_low_violations == []
    # Synthetic GBM data won't naturally produce long stale runs
    assert report.stale_price_runs == []


def test_detects_injected_stale_run():
    df = generate_ohlcv("STALE", n_days=100, seed=7).copy()
    df.loc[df.index[10:16], "close"] = 50.0  # force a frozen price for 6 days
    runs = check_stale_prices(df, min_run_length=4)
    assert len(runs) >= 1
    assert runs[0]["length"] >= 4


def test_detects_injected_split_like_jump():
    df = generate_ohlcv("SPLIT", n_days=100, seed=9).copy()
    # Simulate an unadjusted 2-for-1 split: price halves overnight
    split_idx = 50
    df.iloc[split_idx:, df.columns.get_loc("close")] = df.iloc[split_idx:]["close"] / 2
    suspects = detect_suspected_splits(df, jump_threshold=0.35)
    assert len(suspects) >= 1


def test_adjust_for_suspected_splits_smooths_series():
    df = generate_ohlcv("SPLIT2", n_days=100, seed=11).copy()
    split_idx = 50
    for col in ("open", "high", "low", "close"):
        df.iloc[split_idx:, df.columns.get_loc(col)] = df.iloc[split_idx:][col] / 2

    report = run_quality_checks(df, "SPLIT2")
    assert report.suspected_splits, "test setup should have triggered a suspected split"

    adjusted = adjust_for_suspected_splits(df, report)
    # After adjustment, the series should no longer show the same jump magnitude
    post_adjust_suspects = detect_suspected_splits(adjusted, jump_threshold=0.35)
    assert len(post_adjust_suspects) < len(report.suspected_splits)
