import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from afx_ai.data.synthetic import generate_ohlcv
from afx_ai.features.engineering import build_features, FEATURE_COLUMNS
from afx_ai.pipeline import build_default_ensemble


def test_ensemble_end_to_end():
    df = generate_ohlcv("TEST", n_days=400, seed=3)
    feats = build_features(df)
    X = feats[FEATURE_COLUMNS].values
    y = feats["target_direction"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    ensemble = build_default_ensemble(n_features=X.shape[1])
    ensemble.fit(X_train, y_train)

    proba = ensemble.predict_proba_up(X_test)
    assert proba.shape[0] == X_test.shape[0]
    assert np.all((proba >= 0) & (proba <= 1))

    weights = ensemble.meta_weights()
    assert len(weights) == 4  # gbm, lstm, transformer, stat_arb
