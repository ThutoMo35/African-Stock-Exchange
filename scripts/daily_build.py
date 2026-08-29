#!/usr/bin/env python
"""Daily automated build for the African Stock Exchange AI ensemble system.

Runs the full data -> feature -> ensemble -> backtest pipeline across one
representative ticker per configured exchange, then writes:

  reports/YYYY-MM-DD.json   -- full machine-readable results for that run
  reports/latest.json       -- always points at the most recent run
  reports/latest.md         -- human-readable summary table (for README/CI)
  reports/history.csv       -- one row appended per day, for trend tracking

Intended to be triggered daily by .github/workflows/daily-build.yml, but
runs identically from a local shell:

    python scripts/daily_build.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.config import CONFIG
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"


def _run_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_daily_build(tickers_per_exchange: int = 1) -> dict:
    loader = SyntheticDataLoader(n_days=750)
    run_date = _run_date()
    run_started = datetime.now(timezone.utc).isoformat()

    results = []
    for exchange_code, exchange_cfg in CONFIG.exchanges.items():
        tickers = exchange_cfg["sample_tickers"][:tickers_per_exchange]
        for ticker in tickers:
            print(f"\n[daily-build] {run_date} :: {exchange_code} :: {ticker}")
            try:
                res = run_pipeline(loader, ticker, verbose=False)
                bt = res["backtest"]
                results.append(
                    {
                        "exchange": exchange_code,
                        "ticker": ticker,
                        "n_train": res["n_train"],
                        "n_test": res["n_test"],
                        "meta_weights": res["meta_weights"],
                        "strategy_metrics": bt["metrics"],
                        "buy_hold_metrics": bt["buy_hold_metrics"],
                        "avg_exposure": bt["avg_exposure"],
                        "num_trades": bt["num_trades"],
                        "status": "ok",
                    }
                )
                print(f"  -> sharpe={bt['metrics']['sharpe_ratio']}  "
                      f"cagr={bt['metrics']['cagr']}%  "
                      f"max_dd={bt['metrics']['max_drawdown_pct']}%")
            except Exception as e:  # noqa: BLE001
                print(f"  -> FAILED: {e}")
                results.append(
                    {
                        "exchange": exchange_code,
                        "ticker": ticker,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    report = {
        "run_date": run_date,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(results),
        "n_succeeded": n_ok,
        "n_failed": len(results) - n_ok,
        "results": results,
    }
    return report


def write_report(report: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    run_date = report["run_date"]

    dated_path = REPORTS_DIR / f"{run_date}.json"
    dated_path.write_text(json.dumps(report, indent=2))

    latest_json_path = REPORTS_DIR / "latest.json"
    latest_json_path.write_text(json.dumps(report, indent=2))

    # Human-readable markdown summary
    lines = [
        f"# AFX AI — Daily Build Report ({report['run_date']})",
        "",
        f"Run window (UTC): {report['run_started_utc']} → {report['run_finished_utc']}",
        f"Targets: {report['n_targets']}  |  Succeeded: {report['n_succeeded']}  |  Failed: {report['n_failed']}",
        "",
        "| Exchange | Ticker | Sharpe | CAGR % | Max DD % | Trades | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        if r["status"] == "ok":
            m = r["strategy_metrics"]
            lines.append(
                f"| {r['exchange']} | {r['ticker']} | {m['sharpe_ratio']} | "
                f"{m['cagr']} | {m['max_drawdown_pct']} | {r['num_trades']} | ok |"
            )
        else:
            lines.append(f"| {r['exchange']} | {r['ticker']} | - | - | - | - | FAILED |")
    lines.append("")
    lines.append(
        "_Synthetic-data research pipeline. Not financial advice. See README for scope._"
    )
    (REPORTS_DIR / "latest.md").write_text("\n".join(lines))

    # Append trend row(s) to history.csv
    history_path = REPORTS_DIR / "history.csv"
    is_new = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["run_date", "exchange", "ticker", "sharpe_ratio", "cagr_pct",
                 "max_drawdown_pct", "avg_exposure", "num_trades", "status"]
            )
        for r in report["results"]:
            if r["status"] == "ok":
                m = r["strategy_metrics"]
                writer.writerow(
                    [report["run_date"], r["exchange"], r["ticker"], m["sharpe_ratio"],
                     m["cagr"], m["max_drawdown_pct"], round(r["avg_exposure"], 4),
                     r["num_trades"], "ok"]
                )
            else:
                writer.writerow(
                    [report["run_date"], r["exchange"], r["ticker"], "", "", "", "", "", "failed"]
                )

    print(f"\n[daily-build] Wrote {dated_path.relative_to(REPO_ROOT)}, "
          f"latest.json, latest.md, and appended to history.csv")


def main():
    report = run_daily_build(tickers_per_exchange=1)
    write_report(report)
    if report["n_failed"] > 0:
        print(f"\n[daily-build] WARNING: {report['n_failed']} target(s) failed.")
        sys.exit(1)
    print("\n[daily-build] Completed successfully.")


if __name__ == "__main__":
    main()
