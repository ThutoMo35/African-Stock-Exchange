"""Gradient boosting ensemble member (XGBoost)."""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from afx_ai.config import CONFIG
from afx_ai.models.base import BaseModel


class GradientBoostModel(BaseModel):
    name = "gradient_boost"

    def __init__(self):
        cfg = CONFIG.model
        self.clf = XGBClassifier(
            n_estimators=cfg.gbm_n_estimators,
            max_depth=cfg.gbm_max_depth,
            learning_rate=cfg.gbm_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=cfg.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostModel":
        self.clf.fit(X, y)
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]
