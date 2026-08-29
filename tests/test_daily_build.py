import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from daily_build import parse_todo_progress, evaluate_daily_target


def test_parse_todo_progress_counts_checkboxes(tmp_path):
    todo = tmp_path / "TODO.md"
    todo.write_text(
        "- [x] done one\n"
        "- [X] done two (capital X also counts)\n"
        "- [ ] not done yet\n"
        "- [ ] another pending task\n"
        "some unrelated line\n"
    )
    progress = parse_todo_progress(todo)
    assert progress["total"] == 4
    assert progress["done"] == 2
    assert progress["pct"] == 50.0
    assert progress["next_task"] == "not done yet"


def test_parse_todo_progress_missing_file(tmp_path):
    progress = parse_todo_progress(tmp_path / "does_not_exist.md")
    assert progress == {"total": 0, "done": 0, "pct": 0.0, "next_task": None}


def test_evaluate_daily_target_all_pass():
    report = {
        "n_targets": 2,
        "results": [
            {"status": "ok", "strategy_metrics": {"sharpe_ratio": 1.0, "max_drawdown_pct": -5.0}},
            {"status": "ok", "strategy_metrics": {"sharpe_ratio": 0.8, "max_drawdown_pct": -3.0}},
        ],
    }
    verdict = evaluate_daily_target(report)
    assert verdict["met"] is True
    assert all(c["passed"] for c in verdict["checks"].values())


def test_evaluate_daily_target_fails_on_low_sharpe():
    report = {
        "n_targets": 1,
        "results": [
            {"status": "ok", "strategy_metrics": {"sharpe_ratio": -2.0, "max_drawdown_pct": -5.0}},
        ],
    }
    verdict = evaluate_daily_target(report)
    assert verdict["met"] is False
    assert verdict["checks"]["avg_sharpe"]["passed"] is False
