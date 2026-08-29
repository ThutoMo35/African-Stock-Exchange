# African Stock Exchange AI — Ensemble Intelligence System

[![Daily AFX AI Build](https://github.com/ThutoMo35/African-Stock-Exchange/actions/workflows/daily-build.yml/badge.svg)](https://github.com/ThutoMo35/African-Stock-Exchange/actions/workflows/daily-build.yml)

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

For the full system blueprint (layer-by-layer breakdown, extension points, and the
plan → execute → report → review loop the automation runs on), see **[ARCHITECTURE.md](ARCHITECTURE.md)**.
For the phased roadmap and daily target definition, see **[TODO.md](TODO.md)**.

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

## Automated daily build

`.github/workflows/daily-build.yml` runs the full pipeline automatically every day at
03:00 UTC (and on every push to `main`, and on-demand via **Actions → Daily AFX AI Build →
Run workflow**):

1. Installs dependencies
2. Runs the test suite (`pytest`) as a safety gate — the build stops here if it fails
3. Runs `scripts/daily_build.py`, which retrains the ensemble on one representative ticker
   per configured exchange and backtests out-of-sample
4. Writes results to `reports/`:
   - `reports/YYYY-MM-DD.json` — full result for that day
   - `reports/latest.json` / `reports/latest.md` — most recent run (human + machine readable)
   - `reports/history.csv` — one row per exchange per day, for tracking metrics over time
5. Commits and pushes the updated `reports/` back to the repo automatically
6. Uploads the same report as a downloadable workflow artifact (kept 90 days)

Each run also self-checks against a defined **daily target** (`config.py::DailyTarget`:
minimum average Sharpe, maximum average drawdown, required success rate) and reports
**MET/MISSED** at the top of `reports/latest.md`, logged over time in
`reports/targets_history.csv`. It separately reports roadmap progress by parsing the
checkboxes in `TODO.md`, so every daily report shows exactly what fraction of the plan
is done and what task is next — see `ARCHITECTURE.md` §3 for how this loop fits together.

To change the schedule, edit the `cron` line in the workflow file. To change which/how many
tickers run daily, pass `tickers_per_exchange` to `run_daily_build()` in
`scripts/daily_build.py`.

Run it manually at any time:

```bash
python scripts/daily_build.py
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
