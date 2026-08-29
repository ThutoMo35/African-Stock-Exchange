"""End-to-end orchestration: data -> features -> ensemble -> backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from afx_ai.config import CONFIG
from afx_ai.data.loader import DataLoader
from afx_ai.features.engineering import FEATURE_COLUMNS, build_features
from afx_ai.models.gradient_boost import GradientBoostModel
from afx_ai.models.lstm_model import LSTMModel
from afx_ai.models.statistical import StatArbModel
from afx_ai.models.transformer_model import TransformerModel
from afx_ai.ensemble.stacking import StackingEnsemble
from afx_ai.backtest.engine import run_backtest
from afx_ai.backtest.walkforward import walk_forward_splits, summarize_fold_metrics


def build_default_ensemble(n_features: int) -> StackingEnsemble:
    return StackingEnsemble(
        members=[
            GradientBoostModel(),
            LSTMModel(n_features=n_features),
            TransformerModel(n_features=n_features),
            StatArbModel(),
        ]
    )


def run_pipeline(loader: DataLoader, ticker: str, verbose: bool = True) -> dict:
    cfg = CONFIG.model
    raw = loader.load(ticker)
    feats = build_features(raw)

    X = feats[FEATURE_COLUMNS].values
    y = feats["target_direction"].values
    forward_returns = feats["target_return"].values
    dates = feats.index

    split = int(len(feats) * cfg.train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    fwd_test = forward_returns[split:]
    dates_test = dates[split:]

    ensemble = build_default_ensemble(n_features=X.shape[1])
    ensemble.fit(X_train, y_train)

    proba_test = ensemble.predict_proba_up(X_test)
    bt = run_backtest(dates_test, fwd_test, proba_test)

    result = {
        "ticker": ticker,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "meta_weights": ensemble.meta_weights(),
        "member_breakdown_test": ensemble.member_breakdown(X_test),
        "backtest": bt,
    }

    if verbose:
        print(f"\n=== {ticker} ===")
        print(f"Train samples: {result['n_train']}, Test samples: {result['n_test']}")
        print("Learned meta-learner trust weights (ensemble member -> weight):")
        for name, w in result["meta_weights"].items():
            print(f"  {name:>15s}: {w:+.4f}")
        print("\nStrategy metrics (out-of-sample):", bt["metrics"])
        print("Buy & hold metrics (out-of-sample):", bt["buy_hold_metrics"])
        print(f"Avg exposure: {bt['avg_exposure']:.2%}  |  Trades: {bt['num_trades']}")

    return result


def run_walkforward_pipeline(
    loader: DataLoader, ticker: str, n_folds: int = 5, min_train_size: int = 200, verbose: bool = True
) -> dict:
    """Rolling-origin walk-forward evaluation -- retrains the ensemble across
    several time-ordered folds instead of a single train/test split, then
    reports the mean/std of out-of-sample performance across folds. This is
    the more rigorous alternative used for model validation (Phase 2);
    scripts/daily_build.py still uses the faster single-split
    run_pipeline() to keep the daily CI job quick.
    """
    raw = loader.load(ticker)
    feats = build_features(raw)

    X = feats[FEATURE_COLUMNS].values
    y = feats["target_direction"].values
    forward_returns = feats["target_return"].values
    dates = feats.index

    splits = walk_forward_splits(len(X), n_folds=n_folds, min_train_size=min_train_size)
    if not splits:
        raise ValueError(
            f"Not enough samples ({len(X)}) for {n_folds} walk-forward folds "
            f"with min_train_size={min_train_size}"
        )

    fold_results = []
    for split in splits:
        ensemble = build_default_ensemble(n_features=X.shape[1])
        ensemble.fit(X[split.train_idx], y[split.train_idx])

        proba_test = ensemble.predict_proba_up(X[split.test_idx])
        bt = run_backtest(
            dates[split.test_idx], forward_returns[split.test_idx], proba_test
        )
        fold_results.append(
            {
                "fold": split.fold,
                "train_size": len(split.train_idx),
                "test_size": len(split.test_idx),
                "metrics": bt["metrics"],
            }
        )
        if verbose:
            print(f"  Fold {split.fold}: train={len(split.train_idx)} test={len(split.test_idx)} "
                  f"-> {bt['metrics']}")

    aggregate = summarize_fold_metrics([f["metrics"] for f in fold_results])

    result = {"ticker": ticker, "n_folds": len(fold_results), "folds": fold_results, "aggregate": aggregate}
    if verbose:
        print(f"\n=== {ticker} walk-forward summary across {len(fold_results)} folds ===")
        for k, v in aggregate.items():
            print(f"  {k}: {v}")
    return result
