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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.config import CONFIG
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
TODO_PATH = REPO_ROOT / "TODO.md"

_CHECKBOX_RE = re.compile(r"^\s*-\s*\[( |x|X)\]\s+(.*)$")


def parse_todo_progress(todo_path: Path = TODO_PATH) -> dict:
    """Parse TODO.md's checkboxes into a progress summary, so the plan
    (TODO.md) and the execution report (reports/) stay visibly linked."""
    if not todo_path.exists():
        return {"total": 0, "done": 0, "pct": 0.0, "next_task": None}

    total, done, next_task = 0, 0, None
    for line in todo_path.read_text().splitlines():
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        total += 1
        checked, text = m.group(1).lower() == "x", m.group(2).strip()
        if checked:
            done += 1
        elif next_task is None:
            next_task = text

    pct = round(100 * done / total, 1) if total else 0.0
    return {"total": total, "done": done, "pct": pct, "next_task": next_task}


def evaluate_daily_target(report: dict) -> dict:
    """Score this run against AppConfig.daily_target and return a pass/fail
    verdict with the numbers behind it."""
    target = CONFIG.daily_target
    ok_results = [r for r in report["results"] if r["status"] == "ok"]

    success_rate = len(ok_results) / report["n_targets"] if report["n_targets"] else 0.0
    avg_sharpe = (
        sum(r["strategy_metrics"]["sharpe_ratio"] for r in ok_results) / len(ok_results)
        if ok_results else float("-inf")
    )
    avg_drawdown = (
        sum(r["strategy_metrics"]["max_drawdown_pct"] for r in ok_results) / len(ok_results)
        if ok_results else float("-inf")
    )

    checks = {
        "success_rate": {
            "value": round(success_rate, 4),
            "threshold": target.min_success_rate,
            "passed": success_rate >= target.min_success_rate,
        },
        "avg_sharpe": {
            "value": round(avg_sharpe, 4),
            "threshold": target.min_avg_sharpe,
            "passed": avg_sharpe >= target.min_avg_sharpe,
        },
        "avg_max_drawdown_pct": {
            "value": round(avg_drawdown, 4),
            "threshold": target.max_avg_drawdown_pct,
            "passed": avg_drawdown >= target.max_avg_drawdown_pct,
        },
    }
    overall_met = all(c["passed"] for c in checks.values())
    return {"met": overall_met, "checks": checks}


def _upsert_csv_rows(path: Path, header: list, run_date: str, new_rows: list) -> None:
    """Write new_rows into path, replacing any existing rows whose first
    column equals run_date. Keeps history.csv/targets_history.csv idempotent
    when the daily build runs more than once on the same day."""
    existing_rows = []
    if path.exists():
        with open(path, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            for row in reader:
                if row and row[0] != run_date:
                    existing_rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)
        writer.writerows(new_rows)


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
    report["daily_target"] = evaluate_daily_target(report)
    report["roadmap_progress"] = parse_todo_progress()
    return report


def write_report(report: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    run_date = report["run_date"]

    dated_path = REPORTS_DIR / f"{run_date}.json"
    dated_path.write_text(json.dumps(report, indent=2))

    latest_json_path = REPORTS_DIR / "latest.json"
    latest_json_path.write_text(json.dumps(report, indent=2))

    # Human-readable markdown summary
    target = report["daily_target"]
    roadmap = report["roadmap_progress"]
    verdict = "✅ MET" if target["met"] else "❌ MISSED"

    lines = [
        f"# AFX AI — Daily Build Report ({report['run_date']})",
        "",
        f"**Daily target: {verdict}**",
        "",
        "| Check | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    for name, c in target["checks"].items():
        result = "pass" if c["passed"] else "FAIL"
        lines.append(f"| {name} | {c['value']} | {c['threshold']} | {result} |")

    lines += [
        "",
        f"**Roadmap progress (TODO.md):** {roadmap['done']}/{roadmap['total']} "
        f"tasks complete ({roadmap['pct']}%)",
        f"**Next up:** {roadmap['next_task'] or '_all tracked tasks complete_'}",
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

    # Append trend row(s) to history.csv — replacing any existing row(s) for
    # today's date first, since this script can run more than once per day
    # (push trigger + schedule + manual dispatch all write to the same file).
    history_path = REPORTS_DIR / "history.csv"
    _upsert_csv_rows(
        history_path,
        header=["run_date", "exchange", "ticker", "sharpe_ratio", "cagr_pct",
                "max_drawdown_pct", "avg_exposure", "num_trades", "status"],
        run_date=report["run_date"],
        new_rows=[
            [report["run_date"], r["exchange"], r["ticker"],
             r["strategy_metrics"]["sharpe_ratio"], r["strategy_metrics"]["cagr"],
             r["strategy_metrics"]["max_drawdown_pct"], round(r["avg_exposure"], 4),
             r["num_trades"], "ok"]
            if r["status"] == "ok" else
            [report["run_date"], r["exchange"], r["ticker"], "", "", "", "", "", "failed"]
            for r in report["results"]
        ],
    )

    print(f"\n[daily-build] Wrote {dated_path.relative_to(REPO_ROOT)}, "
          f"latest.json, latest.md, and appended to history.csv")

    # Append to targets_history.csv (tracks pass/fail + roadmap progress over
    # time), replacing any existing row for today's date for the same reason.
    targets_history_path = REPORTS_DIR / "targets_history.csv"
    checks = target["checks"]
    _upsert_csv_rows(
        targets_history_path,
        header=["run_date", "target_met", "avg_sharpe", "avg_max_drawdown_pct",
                "success_rate", "roadmap_done", "roadmap_total", "roadmap_pct"],
        run_date=report["run_date"],
        new_rows=[[
            report["run_date"],
            target["met"],
            checks["avg_sharpe"]["value"],
            checks["avg_max_drawdown_pct"]["value"],
            checks["success_rate"]["value"],
            roadmap["done"],
            roadmap["total"],
            roadmap["pct"],
        ]],
    )
    print(f"[daily-build] Daily target: {'MET' if target['met'] else 'MISSED'}  "
          f"|  Roadmap: {roadmap['done']}/{roadmap['total']} ({roadmap['pct']}%)")


def main():
    report = run_daily_build(tickers_per_exchange=1)
    write_report(report)

    if report["n_failed"] > 0:
        print(f"\n[daily-build] ERROR: {report['n_failed']} target(s) failed to run.")
        sys.exit(1)

    if not report["daily_target"]["met"]:
        print("\n[daily-build] WARNING: daily target MISSED (see reports/latest.md for details).")
        print("[daily-build] Not treated as a hard failure on synthetic data — "
              "tighten config.py::DailyTarget once real data is wired in.")
        # Exit 0 deliberately: a missed *quality* target on synthetic data is
        # expected/normal noise, not a broken pipeline. Only execution errors
        # (handled above) should fail the CI job and page anyone.
        sys.exit(0)

    print("\n[daily-build] Completed successfully. Daily target MET.")


if __name__ == "__main__":
    main()
