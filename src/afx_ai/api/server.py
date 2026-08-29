"""FastAPI serving layer for the ensemble system."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

from afx_ai.config import CONFIG
from afx_ai.data.loader import SyntheticDataLoader
from afx_ai.pipeline import run_pipeline

app = FastAPI(
    title="African Stock Exchange AI — Ensemble Intelligence API",
    description=(
        "Serves ensemble (gradient boosting + LSTM + Transformer + statistical) "
        "predictions across African equity exchanges. Research use only — not "
        "financial advice."
    ),
    version="0.1.0",
)

_loader = SyntheticDataLoader()


class PredictionResponse(PydanticModel):
    ticker: str
    probability_up_next_day: float
    meta_weights: dict
    disclaimer: str = (
        "Research/educational output only. Not financial advice. "
        "Backtested/synthetic performance does not predict future results."
    )


@lru_cache(maxsize=64)
def _cached_pipeline_result(ticker: str):
    return run_pipeline(_loader, ticker, verbose=False)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "African Stock Exchange AI",
        "exchanges": list(CONFIG.exchanges.keys()),
        "docs": "/docs",
    }


@app.get("/exchanges", tags=["meta"])
def list_exchanges():
    return CONFIG.exchanges


@app.get("/predict/{ticker}", response_model=PredictionResponse, tags=["predict"])
def predict(ticker: str):
    try:
        result = _cached_pipeline_result(ticker.upper())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    latest_proba = float(result["backtest"]["equity_curve"].pct_change().iloc[-1]) \
        if len(result["backtest"]["equity_curve"]) else 0.5
    # Use the ensemble's own last test-set probability rather than a return value:
    member_breakdown = result["member_breakdown_test"]
    import numpy as np
    stacked = np.column_stack(list(member_breakdown.values()))
    last_row_proba = float(np.mean(stacked[-1])) if len(stacked) else 0.5

    return PredictionResponse(
        ticker=ticker.upper(),
        probability_up_next_day=round(last_row_proba, 4),
        meta_weights=result["meta_weights"],
    )
