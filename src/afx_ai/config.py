"""Central configuration: exchanges, tickers, model hyperparameters."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
EXCHANGES_YAML = ROOT_DIR / "config" / "exchanges.yaml"


def load_exchanges() -> Dict[str, dict]:
    with open(EXCHANGES_YAML, "r") as f:
        data = yaml.safe_load(f)
    return data.get("exchanges", {})


@dataclass
class ModelConfig:
    lookback_window: int = 30          # trading days of history per sample
    forecast_horizon: int = 1          # days ahead to predict
    train_test_split: float = 0.8
    random_seed: int = 42

    # Gradient boosting
    gbm_n_estimators: int = 300
    gbm_max_depth: int = 5
    gbm_learning_rate: float = 0.05

    # LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_epochs: int = 15
    lstm_lr: float = 1e-3

    # Transformer
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_epochs: int = 15
    transformer_lr: float = 1e-3


@dataclass
class DailyTarget:
    """Defines what 'success' means for a single automated daily run, so the
    build can self-report pass/fail rather than just dumping numbers.

    These thresholds are deliberately conservative starting points for a
    system still running on synthetic data — tighten them once real market
    data is wired in via a custom DataLoader.
    """
    min_avg_sharpe: float = 0.20          # avg out-of-sample Sharpe across the daily universe
    max_avg_drawdown_pct: float = -35.0   # avg max drawdown must not exceed (be more negative than) this
    min_success_rate: float = 1.0         # fraction of exchange targets that must run without error


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    exchanges: Dict[str, dict] = field(default_factory=load_exchanges)
    daily_target: DailyTarget = field(default_factory=DailyTarget)


CONFIG = AppConfig()
