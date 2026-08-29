"""Classical statistical ensemble member: mean-reversion + momentum blend.

No learned parameters beyond a couple of scalars — deliberately included as
a low-variance, interpretable member that doesn't share the LSTM/GBM/
Transformer's failure modes, improving ensemble diversity.
"""
from __future__ import annotations

import numpy as np

from afx_ai.models.base import BaseModel


class StatArbModel(BaseModel):
    name = "stat_arb"

    def __init__(self, rsi_col: int = 6, bb_col: int = 10, momentum_col: int = 13):
        # Column indices correspond to FEATURE_COLUMNS ordering:
        # rsi_14 -> 6, bb_pct_b -> 10, momentum_10 -> 13 (see features/engineering.py)
        self.rsi_col = rsi_col
        self.bb_col = bb_col
        self.momentum_col = momentum_col
        self.mom_weight = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatArbModel":
        # Light calibration: pick the momentum/mean-reversion blend weight
        # that best separates the two classes on training data.
        best_score, best_w = -1.0, 0.5
        for w in np.linspace(0, 1, 11):
            scores = self._raw_score(X, w)
            preds = (scores > np.median(scores)).astype(int)
            acc = (preds == y).mean()
            if acc > best_score:
                best_score, best_w = acc, w
        self.mom_weight = best_w
        return self

    def _raw_score(self, X: np.ndarray, w: float) -> np.ndarray:
        rsi = X[:, self.rsi_col]
        bb = X[:, self.bb_col]
        momentum = X[:, self.momentum_col]

        # Mean-reversion signal: oversold (low RSI, low %B) -> bullish
        mean_reversion = (50 - rsi) / 50 - (bb - 0.5)
        momentum_signal = momentum

        return w * momentum_signal + (1 - w) * mean_reversion

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_score(X, self.mom_weight)
        # squash to (0,1)
        return 1 / (1 + np.exp(-raw * 5))
