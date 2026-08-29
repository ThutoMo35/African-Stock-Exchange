import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afx_ai.data.synthetic import generate_ohlcv
from afx_ai.features.engineering import build_features, FEATURE_COLUMNS


def test_generate_ohlcv_shape():
    df = generate_ohlcv("TEST", n_days=200, seed=1)
    assert len(df) == 200
    assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert (df["high"] >= df["low"]).all()


def test_build_features_no_nans():
    df = generate_ohlcv("TEST", n_days=300, seed=2)
    feats = build_features(df)
    assert not feats[FEATURE_COLUMNS].isna().any().any()
    assert "target_direction" in feats.columns
    assert set(feats["target_direction"].unique()).issubset({0, 1})
