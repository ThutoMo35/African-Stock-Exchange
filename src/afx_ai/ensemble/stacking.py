"""Stacking meta-learner: blends ensemble-member probabilities into a final
signal, learning per-member trust weights rather than a fixed average."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression

from afx_ai.models.base import BaseModel


class StackingEnsemble:
    def __init__(self, members: List[BaseModel]):
        self.members = members
        self.meta_learner = LogisticRegression(max_iter=1000)
        self._fitted = False

    def _member_matrix(self, X: np.ndarray) -> np.ndarray:
        cols = [m.predict_proba_up(X) for m in self.members]
        return np.column_stack(cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        for m in self.members:
            m.fit(X, y)
        meta_X = self._member_matrix(X)
        self.meta_learner.fit(meta_X, y)
        self._fitted = True
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fit() before predicting")
        meta_X = self._member_matrix(X)
        return self.meta_learner.predict_proba(meta_X)[:, 1]

    def member_breakdown(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-member probabilities, useful for explainability / debugging."""
        return {m.name: m.predict_proba_up(X) for m in self.members}

    def meta_weights(self) -> Dict[str, float]:
        """Learned trust weight per ensemble member (from the meta-learner's
        coefficients) — higher magnitude = more influence on final signal."""
        if not self._fitted:
            raise RuntimeError("Fit the ensemble first")
        coefs = self.meta_learner.coef_[0]
        return {m.name: float(c) for m, c in zip(self.members, coefs)}
