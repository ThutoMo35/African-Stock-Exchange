"""Transformer-encoder ensemble member (PyTorch).

Captures longer-range dependencies / regime shifts via self-attention,
complementing the LSTM's recency bias and the GBM's lack of sequence
awareness entirely.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from afx_ai.config import CONFIG
from afx_ai.models.base import BaseModel
from afx_ai.models.lstm_model import _make_sequences


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)  # mean-pool over the sequence dimension
        return self.head(pooled).squeeze(-1)


class TransformerModel(BaseModel):
    name = "transformer"

    def __init__(self, n_features: int, window: int | None = None):
        cfg = CONFIG.model
        self.window = window or cfg.lookback_window
        self.net = _TransformerNet(
            n_features, cfg.transformer_d_model, cfg.transformer_nhead, cfg.transformer_layers
        )
        self.epochs = cfg.transformer_epochs
        self.lr = cfg.transformer_lr
        torch.manual_seed(cfg.random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerModel":
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
        pad = np.full(self.window - 1, 0.5)
        return np.concatenate([pad, probs])
