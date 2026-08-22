# HANDOFF — Next session: **Part 2 — Statistical Foundation Toolkit**

> Read this entire file before doing anything.

---

## 1. Context
Continuing the incremental **Multi-Asset Statistical Arbitrage** project.
- **Part 0 (done):** plan, environment, scaffolding, report skeleton.
- **Part 1 (done — this session):** real data pipeline + universe construction. See below.
- **Your task: Part 2 — Statistical Foundation Toolkit.**

Work **only** on branch `arena/01a029cc-test-project`; commit/push **only** there. Master plan: `PLAN.md`.

---

## 2. What Part 1 delivered (verified, committed, pushed)
- **Real data in-sandbox:** NYSE (Kaggle) dataset — 501 US S&P 500-style securities, daily OHLCV, split-adjusted, GICS sectors, **2010-01-04..2016-12-30** (~7 yr). Cleaned to `data/processed/universe.csv` (851,243 rows) + `symbols.csv` (metadata incl. sector) + `_meta.json`. QA passes **10/10**. `environment/check.sh` now **14/14 PASS**.
- **Pipeline (all under `scripts/`):**
  - `data/download.py` — `--source github` (in-sandbox, default) | `--source yahoo` (**user-run locally** to get the ≥10-yr, split+dividend-adjusted 2010→present panel). Writes checksummed manifest to `data/manifest/`.
  - `data/clean.py` — source-agnostic → uniform long panel (`date,ticker,open,high,low,close,adj_close,volume`).
  - `data/qa.py` — 10 hard checks (prices>0, no NaN, no dupes, span, gaps, volume).
  - `data/run_pipeline.sh` — download→clean→QA→summary; `make data` alias.
  - `analysis/make_data_summary.py` → `results/tables/{data_summary,universe_by_year,sector_counts}.csv` + `results/figures/{universe_by_year,sector_comp}.png`.
  - `plotting/render_report_data.py` → `report/tables/data_section.tex` (auto-generated LaTeX table + figure includes for §Data).
- **Report §Data filled** (`report/report.tex`) with real source/cleaning/adjustment/survivorship content and `\input{tables/data_section.tex}`. `report/build_report.py` now: (1) regenerates figures/tables, (2) compiles LaTeX if a TeX engine exists, (3) always emits `report/report.html` preview (embeds §Data table+figures).
- **Metadata committed:** `scripts/data/sp500_tickers.txt` (505), `sector_map.json` (505), `sp500_symbols.csv`.

### Environment (unchanged from Part 0)
- Reachable: pypi, GitHub API/tarballs. **BLOCKED: Yahoo/stooq/Nasdaq, apt, conda, rust, TeX engines.** pip PEP-668 → use `--user --break-system-packages`.
- Scientific stack installed (numpy/pandas/scipy/statsmodels/matplotlib/seaborn/sklearn/joblib). `cmake` in `~/.local/bin`.
- **Data persistence:** `data/raw`, `data/processed`, `data/manifest/*.json`, `results/*` are git-ignored but DO persist across sessions via the workspace snapshot. If they're missing in a fresh session, re-run `bash scripts/data/run_pipeline.sh --source github`.

---

## 3. YOUR TASK — Part 2: Statistical Foundation Toolkit
Build and **unit-test** the mathematics/statistics layer that later Parts and the report's §Methodology/§Robustness depend on. This is pure Python research tooling (the C++ engine, Parts 3–4, is separate and comes next).

### 3.1 Deliverables
1. **`statkit` package** under `scripts/analysis/statkit/` (or `scripts/statkit/` — pick and document), a small importable package with clean, tested functions:
   - **Stationarity:** ADF test, KPSS test — return statistic + p-value + interpret jointly (opposite nulls). Use `statsmodels.tsa.stattools.adfuller`, `kpss`.
   - **Cointegration:** Engle–Granger two-step (OLS residual + ADF), Johansen test (`statsmodels.tsa.vector_ar.vecm.coint_johansen`).
   - **Residual autocorrelation:** Durbin–Watson (`statsmodels.stats.stattools.durbin_watson`), Ljung–Box (`acorr_ljungbox`).
   - **Heteroskedasticity:** Breusch–Pagan, White tests (`statsmodels.stats.diagnostic`).
   - **HAC correction:** Newey–West standard errors (`statsmodels.regression.linear_model.OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags': ...})`); helper to report t-stat before/after.
   - **Ornstein–Uhlenbeck fit:** discretized OLS regression of Δε_t on ε_{t−1} → θ, μ, σ; implied **half-life** = ln(2)/θ. (This is a research core — later compared to holding period.)
   - **Diebold–Mariano test** for forecast comparison (implement or use a well-tested equivalent).
   - **Multiple-testing:** Bonferroni; **White Reality Check**; **Hansen SPA** (implement a bootstrap-based SPA — this is a key adversarial-critique deliverable, spec §2.3).
   - **Block/stationary bootstrap** for confidence intervals that respect time-series dependence (NOT i.i.d. bootstrap).
2. **Unit tests** on **synthetic data** (spec explicitly permits synthetic data for isolated math tests):
   - OU parameter recovery on simulated OU paths (verify θ̂, half-life to tolerance).
   - ADF/KPSS correct acceptance/rejection on simulated stationary vs. random-walk series.
   - DM test detects superior model on simulated nested/overlapping forecasts.
   - Bootstrap CIs cover the truth at ~95% for known parameters.
   - SPA/White Reality Check: when the best of many random strategies is tested, the corrected p-value is not spuriously significant.
   Use a lightweight test runner (plain `assert` + a `scripts/run_tests.py` that discovers `test_*.py`, OR pytest if you prefer — but keep it dependency-light and runnable via one command).
3. **Wire into the report:** write §Methodology math text where the derivations are requested in `report/report.tex` (§4.1 subsubsections and §4.2) — at minimum full text + equations for OLS, Ridge, Lasso, PCA derivations and the diagnostic-test descriptions. Add a results table only where real data is available (real diagnostics on real residuals come in Part 6 — for now the report text + synthetic-test results are enough).
4. Update `environment/check.sh` to run the statkit unit-test suite (must pass).

### 3.2 Exit gate (must all pass before done)
- [ ] `statkit` imports cleanly; every listed routine exists.
- [ ] Synthetic-data unit tests all pass (run via the documented one-liner).
- [ ] OU parameter recovery + half-life verified on simulated data.
- [ ] ADF & KPSS jointly implemented and tested (opposite nulls stated).
- [ ] Multiple-testing toolkit (Bonferroni + Reality Check/SPA) implemented and unit-tested.
- [ ] Stationary block bootstrap implemented (not i.i.d.).
- [ ] Report §Methodology updated with the required derivations and diagnostic descriptions (equations + prose).
- [ ] `environment/check.sh` passes (14 existing + your new statkit test check = ≥15).
- [ ] `git status` clean after commit; new `HANDOFF.md` written for **Part 3 (C++ Engine Skeleton + Latency Instrumentation)**.

### 3.3 Hints
- This is the "math rigor" foundation — the report's grading is strictest on §2.1 derivations. Write them carefully (full OLS closed-form derivation incl. Gauss–Markov assumptions & their violation in financial series; Ridge as MAP under Gaussian prior; Lasso via subgradient conditions / coordinate descent; PCA via eigendecomposition & the factor-structure link).
- The **OU half-life** and **SPA/Reality Check** are the two most "fund-like" deliverables here — do them well and test them on simulated data so the report can show honest numbers later.
- Keep functions vectorized/pandas-friendly; they'll be called on full panels in Parts 5–10.
- Follow the established convention: tables → `results/tables/`, figures → `results/figures/`, LaTeX fragments → `report/tables/`, all driven from `report/build_report.py`.
- Do **not** implement the strategy backtest in Python here (that's C++, Parts 3–4). This Part is only the statistical toolkit.

---

## 4. Ritual (every Part)
1. Implement → run tests + `environment/check.sh` → update report → commit → push to `arena/01a029cc-test-project`.
2. Overwrite `HANDOFF.md` for the next Part (include what you did, decisions, environment state, next task + exit gate).
3. Keep conventions (paths, build driver, report regeneration). Don't restructure existing files.
