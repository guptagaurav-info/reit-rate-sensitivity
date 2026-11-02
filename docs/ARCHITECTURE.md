# Architecture

```mermaid
flowchart LR
  A[Hugging Face mirror: prices, dividends, Treasury yields] -->|sha256 check| B[ingest.py: subset 24 REITs, SPY, TLT]
  B --> C[data.py: weekly total returns, weekly change in 10y yield]
  C --> D[analysis.py: REIT index on SPY and 10y yield change, Newey-West t]
  D --> E[rate duration by period, year and sector]
  C --> F[analysis.py: walk-forward Treasury hedge, prior 104 weeks only, 5bp costs]
  C --> G[static policy hedge: ratio fixed from 2013-2021, tested 2022-2026]
  E & F & G --> H[pipeline.py: metrics.json, figures]
```

Modules: `ingest.py`, `data.py`, `analysis.py`, `pipeline.py`. Tests: `tests/test_analysis.py` (beta recovery, walk-forward uses past data only, hedge ratio equals the duration ratio, drawdown definition).
