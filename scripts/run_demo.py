#!/usr/bin/env python
"""End-to-end demo: runs the full ensemble pipeline on synthetic data for a
handful of tickers across different African exchanges, printing results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.config import CONFIG
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_pipeline


def main():
    loader = SyntheticDataLoader(n_days=750)
    demo_targets = [
        ("JSE", CONFIG.exchanges["JSE"]["sample_tickers"][0]),
        ("NGX", CONFIG.exchanges["NGX"]["sample_tickers"][0]),
        ("NSE_KE", CONFIG.exchanges["NSE_KE"]["sample_tickers"][0]),
    ]
    for exchange, ticker in demo_targets:
        print(f"\n{'='*60}\nExchange: {exchange}\n{'='*60}")
        run_pipeline(loader, ticker)


if __name__ == "__main__":
    main()
