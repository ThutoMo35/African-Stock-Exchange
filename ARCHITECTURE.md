# AFX AI — System Architecture Blueprint

This document is the system-level blueprint: what each layer does, how data and
control flow through it, what's automated today vs. what's planned (see `TODO.md`),
and the non-functional constraints that shaped the design.

## 1. Design principles

1. **Ensemble over scale.** Robustness comes from combining structurally different,
   independently-erroring models — not from one enormous network. See `README.md`
   for why a literal "500B parameter" model was rejected as a design goal.
2. **Runnable without external dependencies.** Every layer works end-to-end on the
   built-in synthetic data generator, with zero API keys or live feeds required.
   This keeps CI, tests, and the daily automated build deterministic and free to run.
3. **Swap points, not rewrites.** Real data, real brokers, and real deployment
   targets plug into defined interfaces (`DataLoader`, `BaseModel`) rather than
   requiring changes to the pipeline that calls them.
4. **The pipeline reports on itself.** The system doesn't just produce predictions;
   it evaluates whether it's meeting a defined bar (`DailyTarget`) and says so,
   every day, in a artifact anyone can read without running code.

## 2. Layered architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│  AUTOMATION LAYER                                                         │
│  .github/workflows/daily-build.yml                                       │
│  Scheduled (cron) + on-demand + on-push trigger. Runs tests as a gate,   │
│  runs the daily build, commits reports back, uploads artifacts.          │
└───────────────────────────┬───────────────────────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER                                                      │
│  afx_ai.pipeline.run_pipeline()  /  scripts/daily_build.py               │
│  Wires the layers below together for one ticker, or loops across the     │
│  configured exchange universe. Owns train/test split & result assembly.  │
└───────┬─────────────────────────────────────────────────────────┬─────────┘
        ▼                                                         ▼
┌──────────────────────────┐                          ┌──────────────────────────┐
│  DATA LAYER               │                          │  CONFIG / REGISTRY LAYER │
│  afx_ai.data              │                          │  afx_ai.config           │
│  DataLoader (interface)   │                          │  config/exchanges.yaml   │
│   ├─ SyntheticDataLoader  │                          │  ModelConfig, DailyTarget│
│   └─ CSVDataLoader        │                          │  (single source of truth │
│  [future: live feed impl] │                          │   for hyperparameters &  │
└───────────┬────────────────┘                         │   the exchange universe) │
            ▼                                          └──────────────────────────┘
┌──────────────────────────┐
│  FEATURE LAYER            │
│  afx_ai.features          │
│  Technical + statistical  │
│  indicators; target        │
│  construction (direction, │
│  forward return)           │
└───────────┬────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER (ensemble members — afx_ai.models)                          │
│  ┌────────────────┐ ┌───────────┐ ┌─────────────┐ ┌───────────────────┐ │
│  │ GradientBoost   │ │ LSTM      │ │ Transformer │ │ StatArb            │ │
│  │ (XGBoost)       │ │ (PyTorch) │ │ (PyTorch)   │ │ (classical stats)  │ │
│  └────────────────┘ └───────────┘ └─────────────┘ └───────────────────┘ │
│  Common interface: BaseModel.fit(X, y) / predict_proba_up(X)             │
└───────────────────────────┬───────────────────────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  ENSEMBLE / META LAYER                                                    │
│  afx_ai.ensemble.StackingEnsemble                                        │
│  Trains a logistic-regression meta-learner over member probabilities;    │
│  exposes per-member trust weights for explainability.                    │
└───────────────────────────┬───────────────────────────────────────────────┘
                             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  EVALUATION LAYER                                                         │
│  afx_ai.backtest                                                         │
│  Vectorized long/flat backtester; Sharpe, CAGR, max drawdown, hit rate.   │
│  Compares strategy vs. buy-and-hold.                                     │
└───────────┬─────────────────────────────────────────────────┬─────────────┘
            ▼                                                 ▼
┌──────────────────────────┐                      ┌──────────────────────────┐
│  SERVING LAYER            │                      │  OBSERVABILITY LAYER     │
│  afx_ai.api (FastAPI)     │                      │  reports/                │
│  /predict/{ticker}        │                      │  latest.md, latest.json, │
│  /exchanges                │                      │  history.csv,            │
│  On-demand inference       │                      │  targets_history.csv     │
└──────────────────────────┘                      └──────────────────────────┘
```

## 3. The plan → execute → report → review loop

This is the operating loop the whole system runs on, and it's already live, not
aspirational:

```
   PLAN                 EXECUTE                  REPORT                 REVIEW
┌──────────┐        ┌──────────────┐        ┌──────────────────┐     ┌────────────┐
│ TODO.md   │──────▶│ GitHub Action │──────▶│ reports/latest.md │────▶│ Maintainer  │
│ (roadmap, │        │ daily-build.  │        │ + history.csv     │     │ reads report│
│ phases,   │        │ yml: tests →  │        │ + targets_history │     │ checks off  │
│ checkboxes│        │ train ensemble│        │ .csv (MET/MISSED) │     │ TODO items, │
│ + daily   │        │ → backtest →  │        │ committed back to │     │ adjusts     │
│ target    │        │ self-check    │        │ the repo          │     │ DailyTarget,│
│ threshold)│        │ target        │        │                    │     │ opens new   │
└──────────┘        └──────────────┘        └──────────────────┘     │ TODO items  │
     ▲                                                                  └──────┬─────┘
     └──────────────────────────────────────────────────────────────────────┘
                              (loop repeats daily)
```

Concretely:

- **Plan** lives in `TODO.md` — phased, checkbox-tracked, with the daily pass/fail
  bar defined at the bottom of that file and enforced by `config.py::DailyTarget`.
- **Execute** is `scripts/daily_build.py`, triggered by
  `.github/workflows/daily-build.yml` on a schedule, on push, or manually.
- **Report** is the `reports/` directory — dated JSON for machines, Markdown for
  humans, CSV history for trend-watching — auto-committed back to `main`.
- **Review** is a human step: read `reports/latest.md`, decide whether the target
  was met, and update `TODO.md` (check off finished items, add new ones, or
  tighten `DailyTarget` thresholds as real data replaces synthetic data).

## 4. Extension points (where Phase 1+ work plugs in)

| Interface | Current implementation | Where Phase 1+ work attaches |
|---|---|---|
| `afx_ai.data.loader.DataLoader` | `SyntheticDataLoader`, `CSVDataLoader`, `HTTPJSONDataLoader` (generic vendor, mock-tested), `CachedDataLoader` (TTL disk cache wrapper) | Point `HTTPJSONDataLoader` at a real vendor endpoint from an environment with outbound network access; wrap it in `CachedDataLoader` |
| `afx_ai.data.quality` | `run_quality_checks`, `adjust_for_suspected_splits` | Feed real vendor output through this before `build_features()` |
| `afx_ai.models.base.BaseModel` | 4 ensemble members | Add new members (e.g. a sentiment model) without touching the ensemble/pipeline code |
| `afx_ai.ensemble.stacking.StackingEnsemble` | Logistic regression meta-learner | Swap in a gradient-boosted or neural meta-learner; add per-regime weighting |
| `afx_ai.backtest.engine.run_backtest` | Binary long/flat | Extend to variable position sizing, multi-asset portfolios (Phase 3) |
| `afx_ai.backtest.walkforward` | `walk_forward_splits`, `run_walkforward_pipeline()` | Use for model validation before promoting a new ensemble version (Phase 2 champion/challenger) |
| `afx_ai.config.DailyTarget` | Sharpe/drawdown/success-rate thresholds | Tighten thresholds, add new checks (e.g. data-drift) as Phase 4 lands |

## 5. Non-functional constraints

- **Determinism for CI:** synthetic data is seeded per-ticker (`hash(ticker)`), so
  the test suite and demo are reproducible across runs and machines.
- **No secrets in the repo:** the daily workflow uses GitHub's built-in
  `GITHUB_TOKEN` to commit reports — no API keys or credentials are stored in
  this repository. Any future real-data credential must go into GitHub Actions
  encrypted secrets, never into source.
- **Fail loudly, not silently:** `scripts/daily_build.py` exits non-zero if any
  target fails to run, which fails the GitHub Actions job and is visible in the
  Actions tab / commit status — it does not fail silently into a green checkmark.
- **Research scope, explicitly bounded:** nothing in this repository places or
  manages real orders. Phase 5 (paper trading) is scoped to broker sandbox APIs
  only; see `TODO.md`.

## 6. What this is not

- Not a live trading system. No brokerage connectivity exists today.
- Not a single 500B-parameter model — see `README.md` for the reasoning.
- Not investment advice. See the disclaimer in `README.md` and in every API
  response's `disclaimer` field.
