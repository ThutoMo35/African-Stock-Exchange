# African Stock Exchange AI — Ensemble Intelligence System

An artificially intelligent, multi-model **ensemble** system for signal generation and
research across African equity markets (JSE, NGX, NSE Kenya, EGX, GSE, BRVM, and others).

## Honest scope note

This project is deliberately **not** marketed as a "500 billion parameter" model. That figure
does not correspond to anything meaningful for equity time-series prediction:

- No African-market dataset (public or private) has remotely enough tokens/rows to justify
  a model of that scale — you would overfit catastrophically.
- 500B parameters is GPT-4-class LLM territory, built for language, not tabular/time-series
  finance data.
- What actually improves predictive robustness in quant finance is **ensembling diverse,
  independently-erroring model families** — not raw parameter count in a single network.

So instead, this repo implements a genuine **ensemble AI/ML system**: several structurally
different models, each contributing an uncorrelated view, combined by a trained meta-learner
(stacking). That's the architecture that's actually used in serious quant research, and it's
the one built here.

**This is a research/engineering scaffold, not investment advice, and is not connected to a
live brokerage or exchange feed.** Wire in real data feeds and paper-trade extensively before
ever considering live capital.

## Architecture

```
                         ┌─────────────────────────┐
                         │   Feature Engineering    │
                         │ (technical + statistical)│
                         └─────────────┬────────────┘
                                        │
        ┌───────────────┬──────────────┼──────────────┬───────────────┐
        ▼                ▼              ▼              ▼               
┌───────────────┐ ┌─────────────┐ ┌───────────┐ ┌────────────────┐
│ Gradient Boost │ │ LSTM Seq.   │ │Transformer│ │ Statistical /   │
│ (XGBoost/LGBM) │ │  Network    │ │  Encoder  │ │ Mean-Reversion  │
└───────┬────────┘ └──────┬──────┘ └─────┬─────┘ └────────┬────────┘
        │                 │               │                │
        └─────────────────┴───────┬───────┴────────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │   Stacking Meta-Learner     │
                    │ (blends member predictions) │
                    └─────────────┬───────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │  Signal / Position Output   │
                    └───────────────────────────┘
```

### Ensemble members

| Model | Family | What it captures |
|---|---|---|
| `GradientBoostModel` | XGBoost / LightGBM | Non-linear feature interactions, tabular strength |
| `LSTMModel` | Recurrent neural network | Sequential/temporal dependencies in price action |
| `TransformerModel` | Self-attention encoder | Long-range dependencies, regime shifts |
| `StatArbModel` | Classical statistics | Mean-reversion / momentum, no black-box risk |

A `StackingEnsemble` meta-learner (logistic/ridge by default, swappable) learns how much to
trust each member per-regime, rather than a fixed average.

## Supported exchanges (config-driven, extend freely)

- Johannesburg Stock Exchange (JSE) — South Africa
- Nigerian Exchange (NGX) — Nigeria
- Nairobi Securities Exchange (NSE) — Kenya
- Egyptian Exchange (EGX) — Egypt
- Ghana Stock Exchange (GSE) — Ghana
- BRVM — West African regional exchange
- Casablanca Stock Exchange (CSE) — Morocco

See `config/exchanges.yaml`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo on synthetic data (no live feed required)
python scripts/run_demo.py

# Run tests
pytest -q

# Serve predictions over HTTP
uvicorn afx_ai.api.server:app --reload
```

## Plugging in real data

`afx_ai/data/loader.py` defines a `DataLoader` interface. `SyntheticDataLoader` is provided
so the whole pipeline runs out of the box for development/CI. To go live, implement a loader
against your market data vendor (e.g. a broker API, a Bloomberg/Refinitiv feed, or a scraped
exchange bulletin) and point `train.py` / `pipeline.py` at it.

## Project layout

```
src/afx_ai/
  config.py            # exchange + ticker registry
  data/                # data loading (synthetic + interface for real feeds)
  features/            # technical indicator + feature engineering
  models/              # ensemble member models
  ensemble/            # stacking meta-learner
  backtest/            # vectorized backtester + performance metrics
  api/                 # FastAPI serving layer
  pipeline.py           # end-to-end training/inference orchestration
  train.py              # CLI training entrypoint
scripts/run_demo.py     # synthetic end-to-end demo
tests/                  # unit tests
config/exchanges.yaml   # exchange & ticker registry (data-driven)
```

## Disclaimer

For research and educational purposes only. Not financial advice. Past or backtested
performance is not indicative of future results. Verify all data sources and conduct your
own due diligence before deploying capital in any African (or other) equity market.

## License

MIT — see `LICENSE`.
