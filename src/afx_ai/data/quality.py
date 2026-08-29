"""Data quality checks for OHLCV data.

Real market feeds arrive with gaps, stale quotes, and unadjusted stock
splits. This module detects those issues so garbage doesn't silently flow
into feature engineering and models. Referenced from TODO.md Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    ticker: str
    n_rows: int
    missing_business_days: List[str] = field(default_factory=list)
    stale_price_runs: List[dict] = field(default_factory=list)
    suspected_splits: List[dict] = field(default_factory=list)
    negative_or_zero_prices: List[str] = field(default_factory=list)
    high_low_violations: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not any(
            [
                self.missing_business_days,
                self.stale_price_runs,
                self.suspected_splits,
                self.negative_or_zero_prices,
                self.high_low_violations,
            ]
        )

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_rows": self.n_rows,
            "is_clean": self.is_clean,
            "n_missing_business_days": len(self.missing_business_days),
            "n_stale_price_runs": len(self.stale_price_runs),
            "n_suspected_splits": len(self.suspected_splits),
            "n_negative_or_zero_prices": len(self.negative_or_zero_prices),
            "n_high_low_violations": len(self.high_low_violations),
        }


def check_missing_business_days(df: pd.DataFrame) -> List[str]:
    """Flag business days that fall inside the data's date range but have no
    row — a gap that a naive .pct_change() would silently paper over."""
    if df.empty:
        return []
    full_range = pd.bdate_range(df.index.min(), df.index.max())
    missing = full_range.difference(df.index)
    return [d.strftime("%Y-%m-%d") for d in missing]


def check_stale_prices(df: pd.DataFrame, min_run_length: int = 4) -> List[dict]:
    """Flag runs of `min_run_length`+ consecutive identical closing prices —
    a common symptom of a frozen/stale feed rather than real zero volatility."""
    if df.empty or "close" not in df.columns:
        return []
    close = df["close"]
    same_as_prev = close.eq(close.shift(1))
    runs = []
    run_start = None
    for i, is_same in enumerate(same_as_prev):
        if is_same and run_start is None:
            run_start = i - 1
        elif not is_same and run_start is not None:
            run_len = i - run_start
            if run_len >= min_run_length:
                runs.append(
                    {
                        "start": df.index[run_start].strftime("%Y-%m-%d"),
                        "end": df.index[i - 1].strftime("%Y-%m-%d"),
                        "length": run_len,
                        "price": float(close.iloc[run_start]),
                    }
                )
            run_start = None
    if run_start is not None:
        run_len = len(close) - run_start
        if run_len >= min_run_length:
            runs.append(
                {
                    "start": df.index[run_start].strftime("%Y-%m-%d"),
                    "end": df.index[-1].strftime("%Y-%m-%d"),
                    "length": run_len,
                    "price": float(close.iloc[run_start]),
                }
            )
    return runs


def detect_suspected_splits(df: pd.DataFrame, jump_threshold: float = 0.35) -> List[dict]:
    """Flag overnight moves large enough to plausibly be an unadjusted stock
    split/consolidation rather than a genuine price move (a naive heuristic —
    real pipelines should cross-check against a corporate-actions feed)."""
    if df.empty or "close" not in df.columns:
        return []
    ratio = df["close"] / df["close"].shift(1)
    suspects = []
    for date, r in ratio.items():
        if pd.isna(r):
            continue
        if r >= (1 + jump_threshold) or r <= (1 - jump_threshold):
            suspects.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ratio": round(float(r), 4),
                    "likely_split_factor": round(float(1 / r), 2) if r > 0 else None,
                }
            )
    return suspects


def check_price_sanity(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Flag non-positive prices and high < low violations."""
    negative_or_zero, high_low_violations = [], []
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            continue
        bad = df.index[df[col] <= 0]
        negative_or_zero.extend(d.strftime("%Y-%m-%d") for d in bad)

    if {"high", "low"}.issubset(df.columns):
        bad = df.index[df["high"] < df["low"]]
        high_low_violations.extend(d.strftime("%Y-%m-%d") for d in bad)

    return sorted(set(negative_or_zero)), sorted(set(high_low_violations))


def run_quality_checks(df: pd.DataFrame, ticker: str) -> QualityReport:
    negative_or_zero, high_low_violations = check_price_sanity(df)
    return QualityReport(
        ticker=ticker,
        n_rows=len(df),
        missing_business_days=check_missing_business_days(df),
        stale_price_runs=check_stale_prices(df),
        suspected_splits=detect_suspected_splits(df),
        negative_or_zero_prices=negative_or_zero,
        high_low_violations=high_low_violations,
    )


def adjust_for_suspected_splits(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """Best-effort split adjustment: for each suspected split date, scale all
    prices before that date by the detected ratio so the series is
    continuous. This is a heuristic fallback — a real corporate-actions feed
    should always be preferred when available."""
    if not report.suspected_splits:
        return df
    adjusted = df.copy()
    for split in sorted(report.suspected_splits, key=lambda s: s["date"]):
        split_date = pd.Timestamp(split["date"])
        factor = split.get("likely_split_factor")
        if not factor or factor <= 0:
            continue
        mask = adjusted.index < split_date
        for col in ("open", "high", "low", "close"):
            if col in adjusted.columns:
                adjusted.loc[mask, col] = adjusted.loc[mask, col] / factor
    return adjusted
