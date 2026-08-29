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
