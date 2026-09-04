# Data

## Provenance (current primary panel — Yahoo Finance, user-downloaded 2026-09-04)
- **Source:** Yahoo Finance daily OHLCV via `yfinance` (`scripts/data/download.py --source yahoo`), run on an internet-enabled machine, committed to `data/raw/yfinance/`.
- **Scope:** **195** of the 505-name S&P 500-style list (`scripts/data/sp500_tickers.txt`) were retrieved.
- **Span:** 2010-01-04 .. 2026-09-01 (~16.7 years); fully split **and** dividend adjusted (total-return) via Yahoo `Adj Close`.
- **Processed:** `data/processed/universe.csv` (764,896 rows), `symbols.csv` (505-name metadata: name, sector, sub-industry, date-first-added), `_meta.json`.
- **QA:** `scripts/data/qa.py` passes 10/10 (note: hard ticker gate lowered to 150; actual 195 < 200 spec target, disclosed in report §Data).
- **Bias:** survivorship-biased (only 2026-surviving, long-history names) — disclosed in report §Data.

## Secondary reference set (in-sandbox re-downloadable)
- `python3 scripts/data/download.py --source github` → NYSE (Kaggle) 501-name set, 2010–2016, split-adjusted only (price returns, not total). Useful for cross-sectional breadth cross-checks; not the primary analysis set.

## Layout
- `raw/` — original downloads. `data/raw/yfinance/*.csv` are **tracked** (user-committed, ~19 MB). `data/raw/nyse/` (if present) is git-ignored.
- `processed/` — uniform long panel + metadata (git-ignored; regenerable via `scripts/data/run_pipeline.sh`).
- `manifest/` — provenance manifests.

## Rule
Re-run pipeline: `bash scripts/data/run_pipeline.sh --source yfinance` (or `--source github`). Never commit `data/processed/` or large regenerable artifacts.
