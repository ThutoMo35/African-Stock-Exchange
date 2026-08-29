"""Walk-forward (rolling-origin) cross-validation.

A single train/test split (used by the fast daily build for speed) tells
you how the model does on one slice of history. Walk-forward evaluation
retrains across several rolling windows and reports the distribution of
out-of-sample performance -- much harder to fool with a lucky split.
Referenced from TODO.md Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_size: int = 200,
    test_size: int | None = None,
    expanding: bool = True,
) -> List[WalkForwardSplit]:
    """Generate rolling-origin folds over a time-ordered dataset of length
    n_samples. Each fold's test window comes strictly after its train
    window -- no look-ahead leakage.

    expanding=True: each fold's training set grows (anchored start, like a
        real system that accumulates more history over time).
    expanding=False: each fold's training set is a fixed-size rolling window.
    """
    if test_size is None:
        remaining = n_samples - min_train_size
        test_size = max(1, remaining // n_folds)

    splits = []
    train_end = min_train_size
    for fold in range(n_folds):
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples:
            break

        train_start = 0 if expanding else max(0, train_end - min_train_size)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) == 0:
            break

        splits.append(WalkForwardSplit(train_idx=train_idx, test_idx=test_idx, fold=fold))
        train_end = test_end

    return splits


def summarize_fold_metrics(fold_metrics: List[dict]) -> dict:
    """Aggregate per-fold backtest metrics into mean/std across folds --
    the number that actually matters for judging robustness, not any single
    fold's result."""
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    summary = {}
    for k in keys:
        values = [m[k] for m in fold_metrics if k in m]
        summary[f"{k}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{k}_std"] = round(float(np.std(values)), 4)
    return summary
