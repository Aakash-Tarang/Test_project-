# Data

## Provenance (current in-sandbox dataset — populated by Part 1)
- **Source:** NYSE (Kaggle) dataset — `ashishpatel26/NYSE-STOCK_MARKET-ANALYSIS-USING-LSTM` (GitHub), downloaded as a repo tarball via `api.github.com/repos/.../tarball/HEAD`.
- **Files:** `prices-split-adjusted.csv` (daily OHLCV), `securities.csv` (GICS sector/sub-industry, date first added), `fundamentals.csv` (quarterly).
- **Scope:** 501 U.S. securities, daily, **2010-01-04 .. 2016-12-30** (~7 years), 851,243 cleaned rows.
- **Adjustment:** split-adjusted only (no dividend adjustment) → price (split-adjusted) returns, not total returns.
- **Retrieval:** re-run `python3 scripts/data/download.py --source github` to re-fetch; manifest/checksums in `data/manifest/github_nyse.json`.
- **Bias:** 2016-era constituent snapshot → survivorship bias + short span disclosed in report §Data.

## Recommended ≥10-year primary panel (run on a machine with normal internet)
- `python3 scripts/data/download.py --source yahoo` fetches daily OHLCV + fully-adjusted Adj Close from Yahoo Finance for the 505-ticker S&P 500 list (`scripts/data/sp500_tickers.txt`) over 2010→present into `data/raw/yfinance/`.
- Then `clean.py --source yfinance`, `qa.py`, and `make_data_summary.py` run identically.

## Layout
- `raw/` — original downloads (git-ignored).
- `processed/` — `universe.csv` (long panel: date,ticker,OHLCV,adj_close,volume), `symbols.csv` (metadata), `_meta.json` (provenance note).
- `manifest/` — checksummed provenance manifests (`*.json` regenerable; `*.md` tracked).

## Rule
Every raw file traces to a manifest entry. Never commit raw data or large binaries.
