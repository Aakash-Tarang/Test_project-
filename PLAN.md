# Master Execution Plan — Multi-Asset Statistical Arbitrage

**Project:** Basket statistical arbitrage via regularized regression, nonlinear extensions, and a low-latency C++ execution engine.
**Execution model:** Incremental, multi-session. Each session completes exactly **one Part** of this plan, commits it, and writes a `HANDOFF.md` that fully briefs the next session. Every Part has a hard **Exit Gate** (preliminary checks) that must pass before that Part is declared done.

**Golden rule:** No number appears in the report unless it was produced by the checked-in code in this repository. Nothing is reported without a significance test, out-of-sample validation, and cost adjustment. A negative/robust-negative result, clearly explained, is a success.

---

## 0. Environment Constraints (verified — RE-READ BEFORE EVERY SESSION)

These were empirically confirmed on 2026-08-22 and will be re-verified by `environment/check.sh` each session.

| Resource | Status | Consequence |
|---|---|---|
| `g++` 12.2 + `make` | ✅ available | C++ engine builds |
| `cmake` 4.4.2 | ✅ via `python3 -m pip install --user cmake` (in `~/.local/bin`) | CMake build works |
| Python 3.11 + pip | ✅ | Sci stack installed: numpy/pandas/scipy/statsmodels/matplotlib/seaborn/sklearn/joblib/yfinance/reportlab |
| pip | ⚠️ PEP-668 externally-managed | MUST use `--user --break-system-packages` |
| pypi.org, files.pythonhosted.org | ✅ reachable | Python deps install fine |
| **github.com, api.github.com, codeload.github.com** | ✅ reachable | **Real data source** (repo tarballs) |
| **All market-data APIs** (Yahoo, stooq, Nasdaq, Google) | ❌ **blocked** | Data MUST come from GitHub-hosted datasets |
| Debian apt mirrors, conda, rust, crates.io | ❌ blocked | No `sudo apt install texlive` etc. |
| **LaTeX engine (pdflatex/tectonic/etc.)** | ❌ **not installable** | See §0.1 report strategy |
| CPU / RAM | 2 cores / 3.8 GB | Keep datasets < ~1 GB; parallelize conservatively |

### 0.1 Report (LaTeX) strategy — critical
No TeX engine can be installed in this sandbox. **The deliverable is still a canonical, fully-compilable LaTeX report** (`report/report.tex`), but:
1. The LaTeX source is the *source of truth* — written to compile cleanly in any normal TeX environment (`latexmk -pdf` / `pdflatex`/`biber`).
2. `report/build_report.py` (the single report entry point) does three things in order:
   - **(a)** regenerates every figure and table from the actual pipeline outputs (CSV/JSON in `results/`);
   - **(b)** if a TeX engine is present, compiles `report.tex` → `report.pdf`;
   - **(c)** **always** renders a self-contained `report/report.html` working preview (matplotlib-rendered math images so no CDN is needed) so the report is inspectable in-browser in this sandbox.
3. The PDF is the canonical deliverable when built in a TeX-capable environment; the HTML is the sandbox preview. Both derive from the same content.

---

## 1. Part Breakdown (13 parts, ~10–15 sessions)

Each Part = one session. Deliverables per Part, in order: (1) implement the chunk, (2) update `report/report.tex` and regenerate figures/tables, (3) run `environment/check.sh` + full exit gate, (4) commit, (5) push, (6) write `HANDOFF.md` for the next Part.

### Part 1 — Data Acquisition & Universe Construction
**Goal:** obtain and curate a real, documented equity dataset.
- Acquire daily OHLCV for a large US equity universe (≥200 names, target: S&P 500) **from a GitHub-hosted dataset** (tarball via `api.github.com/repos/{owner}/{repo}/tarball/{ref}`). Candidate sources identified in `environment/data_candidates.md`.
- Build universe from S&P 500 constituent history (survivorship-bias-free if possible; use a defined historical snapshot otherwise and *disclose* the bias quantitatively).
- Pipeline: `scripts/data/download.py`, `scripts/data/clean.py` → parquet/CSV in `data/processed/` (excluded from git via `.gitignore`).
- **Exit gate:** ≥200 names × ≥10 years daily; corporate actions (splits/dividends) documented & handled (Adj Close); no lookahead; a data summary table written to `results/data_summary.csv`; survival-bias analysis written to report §Data.

### Part 2 — Statistical Foundation Toolkit
**Goal:** build + unit-test the math/statistics layer in Python (used by scripts and by C++ cross-checks).
- Implement: ADF & KPSS; Engle–Granger; Johansen (via `statsmodels`); Durbin–Watson; Ljung–Box; Breusch–Pagan/White; Newey–West HAC; OU fit (discretized regression) + half-life; DM test; Bonferroni + White Reality Check + Hansen SPA; block bootstrap (stationary bootstrap).
- Wrap as a `statkit` package with unit tests on synthetic data (verify OU parameter recovery, test null distributions).
- **Exit gate:** every statistical routine has a passing unit test on synthetic data; results are reproducible; `statkit` imports cleanly.

### Part 3 — C++ Engine Skeleton + Latency Instrumentation
**Goal:** compiler-verified, dependency-free, latency-aware C++ core with the required architecture.
- Implement `MarketDataBuffer` (SoA ring buffer), `RollingRegression` (Welford/rank-1 incremental `XᵀX`), `SignalGenerator`, `PortfolioBook`, `BacktestEngine` (interface + latency histograms).
- Eigen: **vendored** single-header `third_party/eigen/` (see §0.1 note below) OR hand-rolled linear algebra — decision deferred to Part 3, but must be **dependency-free at build time** (no `#include` that isn't in the repo or toolchain).
- Hand-rolled test harness (`test/minimal_test.hpp` + `RUN_TEST`) so no external test framework is required.
- Latency instrumentation (p50/p95/p99 histograms per stage) from the start.
- **Exit gate:** CMake build passes; every class has a unit test; a `perf`/cycle-count microbenchmark (`bench/`) runs; zero dynamic allocation verified in the hot path (allocation-tracking test); `--version` smoke run.

### Part 4 — Correctness Core of the Engine
**Goal:** make the simulation loop correct, walk-forward, cost-aware, lookahead-free.
- Implement walk-forward backtest loop, next-bar execution (no lookahead), spread-based cost model + square-root market impact + borrow cost.
- Add delisting-handling, window-not-yet-full edge cases; brute-force vs. incremental regression cross-check.
- **Exit gate:** incremental regression matches brute force to 1e-8; no-lookahead property tested; gross vs. net P&L both computed; edge-case unit tests pass.

### Part 5 — Baseline Model Comparison (OLS / Ridge / Lasso / Elastic Net / PCA)
**Goal:** first real results. Implement + benchmark all regularized/penalized regression variants for basket selection.
- Extend `BasketSelector` to OLS/Ridge/Lasso/ElasticNet/PCA (coordinate descent for Lasso path; eigendecomposition for PCA).
- Report in/out-of-sample R², coefficient stability over time, compute cost.
- **Exit gate:** all 5 linear variants produce results vs. a common pipeline; OOS R² and stability time-series plots generated; section updated in report.

### Part 6 — Statistical Diagnostics on the Final Model(s)
**Goal:** run every test in spec §2.2 on the chosen residuals and report honestly.
- Stationarity, cointegration, autocorrelation, heteroskedasticity, HAC correction (show before/after t-stats), OU half-life vs. holding-period.
- **Exit gate:** all §2.2 diagnostics run and tabulated for the final model(s); half-life-vs-holding-period mismatch explicitly discussed; multiple-testing correction (§2.3) with a numeric answer.

### Part 7 — Portfolio-Level Construction + Risk Controls
**Goal:** a *book*, not a single strategy.
- Multiple concurrent basket trades, inverse-vol / risk-parity sizing, VaR/CVaR, gross/net limits, sector-neutrality — enforced and *verified* in the backtest.
- Regime segmentation (VIX terciles) + stress-period performance (2008, COVID-2020, 2022).
- **Exit gate:** portfolio risk controls are enforced and verified; regime/stress table produced; sensitivity heatmaps (lookback × λ, z-threshold) generated.

### Part 8 — Kalman-Filtered Time-Varying Beta
**Goal:** genuine adaptive estimation vs. fixed-window OLS.
- State-space model (`β_t = β_{t-1} + η_t`, `Y_t = X_tβ_t + ε_t`); tracking error + OOS comparison.
- **Exit gate:** Kalman implemented in C++ (or Python reference + C++ port), compared to rolling OLS; report section updated.

### Part 9 — Nonlinear Extension + RESET Test
**Goal:** honestly evaluate whether nonlinear signal extraction adds value.
- Gradient-boosted trees (Python, sklearn) / shallow NN; learning curves; Ramsey RESET test; bias–variance discussion.
- **Exit gate:** nonlinear vs. linear OOS R² and Sharpe compared with honest conclusion; learning curves + RESET plotted; report section updated.

### Part 10 — Multiple-Testing Correction + Diebold–Mariano + Final Statistical Validity
**Goal:** the adversarial self-critique numbers.
- Full hypothesis-count audit, Bonferroni, White Reality Check, Hansen SPA, DM test across model pairs (forest plot).
- **Exit gate:** explicit numeric answer to "is the best Sharpe still distinguishable from zero after correction?"; DM results table; section 8 complete.

### Part 11 — System Architecture & Latency Analysis (Report Section 9)
**Goal:** full architecture writeup + latency/cache benchmarks in the report.
- Vector vs. deque, SoA vs. AoS microbenchmarks; latency histograms (p50/p95/p99) plotted; design rationale (cache lines, prefetch, no-alloc, CRTP).
- **Exit gate:** all §6.1 latency/cache plots generated from real runs; section 9 written; benchmark numbers traceable.

### Part 12 — Full Report Assembly + Non-Goals
**Goal:** compile the complete report with all sections, figures, tables, references; write the "What This Does Not Replicate" section; abstract; conclusion.
- **Exit gate:** `report/build_report.py` produces a complete, internally-consistent report (PDF when a TeX engine is present; HTML preview otherwise) with every required section and plot present and traced to results.

### Part 13 — Final Review, Reproducibility, and Delivery
**Goal:** end-to-end verification + README.
- Full exit-gate sweep, reproducibility run, final README with exact reproduction commands, deliverable checklist from spec §7 verified line-by-line.
- **Exit gate:** clean end-to-end run; every checklist item ticked; final commit + tag; overview/summary added to the report.

---

## 2. Cross-Cutting Rules

- **Data placement:** raw data lives in `data/` (git-ignored). `scripts/data/` has the downloader + a checksum manifest. Never commit raw market data.
- **Results placement:** every number/plot lives in `results/` (git-ignored except small tracked summaries/tables). Figures embed a "generated by `<script>` on `<timestamp>` from `<input>`" tag.
- **Report is built, not hand-written for numbers:** `report/build_report.py` regenerates figures/tables from `results/`. No manually transcribed numbers.
- **C++ is the simulation core; Python is the research/plotting layer.** No strategy simulation logic in Python.
- **No lookahead, walk-forward only, always report gross AND net-of-cost.**
- **Significance or it didn't happen:** no result without its test; no claim without its multiple-testing-corrected companion.
- **Commit hygiene:** each Part = one or a few logical commits on `arena/01a029cc-test-project`; never commit to another branch; push only to this branch.

---

## 3. Exit Gate — run before finishing EVERY Part
Run `environment/check.sh` (must pass) then verify:
1. Everything I implemented compiles / imports and passes its unit tests.
2. Any figure/table referenced in the report exists in `results/` and was produced by code in the repo.
3. The report section(s) touched by this Part are updated and build via `report/build_report.py`.
4. No raw market data or large artifacts are committed.
5. `git status` is clean after commit; `HANDOFF.md` written for the next Part.

---

## 4. Open Decisions to resolve in Part 1/3 (documented, not blocking)
- **Exact GitHub data source** for daily OHLCV (top candidates in `environment/data_candidates.md`; Part 1 selects after QA).
- **Eigen vs. hand-rolled linear algebra** for the C++ engine (Part 3 decides; must be dependency-free).
- **LaTeX engine availability** — report must always be buildable as canonical LaTeX; HTML is the sandbox preview.
