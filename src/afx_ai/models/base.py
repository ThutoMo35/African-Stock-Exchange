"""Abstract base class every ensemble member implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        """Return P(next-day up move) for each row of X, shape (n,)."""
        ...
