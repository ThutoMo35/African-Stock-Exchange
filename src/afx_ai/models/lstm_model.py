"""LSTM sequence-model ensemble member (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from afx_ai.config import CONFIG
from afx_ai.models.base import BaseModel


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _make_sequences(X: np.ndarray, window: int) -> np.ndarray:
    """Turn a (n, features) matrix into (n - window + 1, window, features)
    sliding-window sequences."""
    n = X.shape[0]
    if n < window:
        raise ValueError(f"Need at least {window} rows to build sequences, got {n}")
    seqs = np.stack([X[i : i + window] for i in range(n - window + 1)])
    return seqs


class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, n_features: int, window: int | None = None):
        cfg = CONFIG.model
        self.window = window or cfg.lookback_window
        self.net = _LSTMNet(n_features, cfg.lstm_hidden_size, cfg.lstm_num_layers)
        self.epochs = cfg.lstm_epochs
        self.lr = cfg.lstm_lr
        torch.manual_seed(cfg.random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMModel":
        seqs = _make_sequences(X, self.window)
        targets = y[self.window - 1 :]

        X_t = torch.tensor(seqs, dtype=torch.float32)
        y_t = torch.tensor(targets, dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self.net(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        seqs = _make_sequences(X, self.window)
        with torch.no_grad():
            logits = self.net(torch.tensor(seqs, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy()
        # pad the warmup rows (no sequence available yet) with 0.5 (neutral)
        pad = np.full(self.window - 1, 0.5)
        return np.concatenate([pad, probs])
