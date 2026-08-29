# AFX AI — Roadmap & TODO

This file is the project's living plan. The daily automation
(`scripts/daily_build.py`) parses the checkboxes below to compute roadmap
progress and surface the next unstarted task in every daily report — so this
file is not just documentation, it's read by the pipeline itself.

Check an item by changing `- [ ]` to `- [x]` when it's done and merged.

---

## Phase 0 — Foundation (done)

- [x] Ensemble architecture: gradient boosting + LSTM + Transformer + statistical member
- [x] Stacking meta-learner
- [x] Feature engineering pipeline (technical + statistical indicators)
- [x] Synthetic data generator + pluggable `DataLoader` interface
- [x] Vectorized backtester with Sharpe / CAGR / drawdown metrics
- [x] FastAPI serving layer
- [x] Config-driven exchange registry (JSE, NGX, NSE Kenya, EGX, GSE, BRVM, CSE Morocco)
- [x] Test suite
- [x] Daily automated build via GitHub Actions (scheduled + on-demand)
- [x] System architecture blueprint (`ARCHITECTURE.md`)
- [x] Daily target / self-check metric wired into the automated build

## Phase 1 — Real data integration

- [x] Generic vendor HTTP/JSON `DataLoader` implemented with retry/backoff, unit-tested against mocked responses (`src/afx_ai/data/vendor.py`, `tests/test_vendor_loader.py`)
- [x] Add data quality checks (missing bars, stale prices, corporate action / stock-split detection) — `src/afx_ai/data/quality.py`
- [x] Add a caching layer so repeated daily runs don't re-fetch unchanged history — `src/afx_ai/data/cache.py`
- [ ] Extend `config/exchanges.yaml` tickers from a handful of samples to a full listed-universe pull
- [ ] Add fundamental data features (earnings, P/E, sector) alongside technical features
- [ ] Live validation against a real JSE/NGX/NSE vendor endpoint — **blocked**: this build environment's network egress is restricted to package registries and GitHub, so it cannot reach financial data vendors. Point `HTTPJSONDataLoader.base_url` at a real vendor and run from an environment with outbound access to validate end-to-end.

## Phase 2 — Model quality & validation

- [x] Replace the single train/test split with walk-forward (rolling-origin) cross-validation — `src/afx_ai/backtest/walkforward.py`, `run_walkforward_pipeline()`. (The daily automated build still uses the faster single-split `run_pipeline()` to keep CI fast; walk-forward is available for deeper validation runs.)
- [ ] Add hyperparameter search (Optuna or similar) for each ensemble member
- [ ] Track per-member and blended accuracy/AUC over time, not just Sharpe
- [ ] Add feature importance / SHAP explainability reporting per model
- [ ] Add a champion/challenger framework so new model versions must beat the current one before promotion

## Phase 3 — Risk & portfolio construction

- [ ] Position sizing beyond binary long/flat (e.g. Kelly-fraction or volatility-targeted sizing)
- [ ] Portfolio-level construction across multiple tickers/exchanges (correlation-aware)
- [ ] Stop-loss / max-drawdown circuit breakers in the backtester
- [ ] Currency-risk handling across exchanges (ZAR, NGN, KES, EGP, GHS, XOF, MAD)

## Phase 4 — Monitoring & observability

- [ ] Alerting when a daily run's Sharpe/drawdown breaches the target thresholds (`config.py::DailyTarget`)
- [ ] Dashboard (Grafana or a simple static site) rendered from `reports/history.csv`
- [ ] Data drift detection on incoming features vs training distribution
- [ ] Model staleness detection (auto-flag when a model hasn't been retrained in N days)

## Phase 5 — Paper trading & deployment

- [ ] Paper-trading harness against a broker sandbox API (no real capital)
- [ ] Daily paper P&L reconciliation report
- [ ] Containerize the API (`docker/Dockerfile`) and document a deployment target
- [ ] Rate limiting / auth on the FastAPI serving layer before any external exposure

## Phase 6 — Documentation & governance

- [ ] Model card per ensemble member (assumptions, known failure modes, training data description)
- [ ] Contribution guide (`CONTRIBUTING.md`)
- [ ] Changelog (`CHANGELOG.md`) tied to releases
- [ ] Expanded compliance/disclaimer language reviewed against relevant securities regulations per exchange jurisdiction

---

## Daily target definition

Every automated run is scored against `AppConfig.daily_target` in `src/afx_ai/config.py`:

| Check | Threshold |
|---|---|
| Average out-of-sample Sharpe across the daily universe | ≥ 0.20 |
| Average max drawdown across the daily universe | ≥ -35.0% (i.e. not worse) |
| Fraction of exchange targets that ran without error | ≥ 100% |

Tighten these once real market data replaces the synthetic generator — the current
thresholds are deliberately loose starting points, not a claim of live trading quality.

The result of this check appears at the top of `reports/latest.md` every day as
**MET** or **MISSED**, and is logged per-day in `reports/targets_history.csv`.
