import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.backtest.walkforward import walk_forward_splits, summarize_fold_metrics
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_walkforward_pipeline


def test_walk_forward_splits_no_lookahead_leakage():
    splits = walk_forward_splits(n_samples=1000, n_folds=5, min_train_size=200)
    assert len(splits) == 5
    for s in splits:
        assert s.train_idx.max() < s.test_idx.min(), "test fold must come strictly after train fold"
    # folds should walk forward, not overlap in test windows
    for a, b in zip(splits, splits[1:]):
        assert a.test_idx.max() < b.test_idx.min()


def test_walk_forward_expanding_window_grows():
    splits = walk_forward_splits(n_samples=1000, n_folds=4, min_train_size=200, expanding=True)
    train_sizes = [len(s.train_idx) for s in splits]
    assert train_sizes == sorted(train_sizes), "expanding window should never shrink"


def test_summarize_fold_metrics():
    fold_metrics = [
        {"sharpe_ratio": 1.0, "cagr": 10.0},
        {"sharpe_ratio": -0.5, "cagr": -5.0},
        {"sharpe_ratio": 0.5, "cagr": 5.0},
    ]
    summary = summarize_fold_metrics(fold_metrics)
    assert summary["sharpe_ratio_mean"] == round((1.0 - 0.5 + 0.5) / 3, 4)
    assert "sharpe_ratio_std" in summary


def test_run_walkforward_pipeline_end_to_end():
    loader = SyntheticDataLoader(n_days=900)
    result = run_walkforward_pipeline(loader, "TEST", n_folds=3, min_train_size=300, verbose=False)
    assert result["n_folds"] == 3
    assert "sharpe_ratio_mean" in result["aggregate"]
    assert len(result["folds"]) == 3
