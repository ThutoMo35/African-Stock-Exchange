"""CLI training entrypoint.

Usage:
    python -m afx_ai.train --ticker NPN --exchange JSE
    python -m afx_ai.train --all-sample-tickers --exchange NGX
"""
from __future__ import annotations

import argparse

from afx_ai.config import CONFIG
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train the AFX ensemble on one or more tickers")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker symbol")
    parser.add_argument("--exchange", type=str, default="JSE", help="Exchange code from config/exchanges.yaml")
    parser.add_argument("--all-sample-tickers", action="store_true", help="Run every sample ticker for the chosen exchange")
    args = parser.parse_args()

    loader = SyntheticDataLoader()
    exchange_cfg = CONFIG.exchanges.get(args.exchange)
    if exchange_cfg is None:
        raise SystemExit(f"Unknown exchange '{args.exchange}'. Options: {list(CONFIG.exchanges)}")

    if args.all_sample_tickers:
        tickers = exchange_cfg["sample_tickers"]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [exchange_cfg["sample_tickers"][0]]

    for t in tickers:
        run_pipeline(loader, t)


if __name__ == "__main__":
    main()
