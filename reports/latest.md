# AFX AI — Daily Build Report (2026-08-29)

**Daily target: ❌ MISSED**

| Check | Value | Threshold | Result |
|---|---|---|---|
| success_rate | 1.0 | 1.0 | pass |
| avg_sharpe | -0.2796 | 0.2 | FAIL |
| avg_max_drawdown_pct | -18.1757 | -35.0 | pass |

**Roadmap progress (TODO.md):** 11/37 tasks complete (29.7%)
**Next up:** Implement a `DataLoader` against one real vendor/exchange feed (start with one exchange, e.g. JSE)

Run window (UTC): 2026-08-29T22:31:15.674564+00:00 → 2026-08-29T22:31:49.838804+00:00
Targets: 7  |  Succeeded: 7  |  Failed: 0

| Exchange | Ticker | Sharpe | CAGR % | Max DD % | Trades | Status |
|---|---|---|---|---|---|---|
| JSE | NPN | -2.947 | -53.49 | -39.24 | 57 | ok |
| NGX | DANGCEM | -0.26 | -7.97 | -16.68 | 62 | ok |
| NSE_KE | SCOM | 1.063 | 19.94 | -9.95 | 45 | ok |
| EGX | COMI | 0.855 | 17.28 | -15.36 | 47 | ok |
| GSE | MTNGH | 0.053 | -4.07 | -9.26 | 54 | ok |
| BRVM | SNTS | 0.593 | 3.09 | -21.44 | 44 | ok |
| CSE_MA | IAM | -1.314 | -18.92 | -15.3 | 47 | ok |

_Synthetic-data research pipeline. Not financial advice. See README for scope._