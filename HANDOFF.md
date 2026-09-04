# HANDOFF — Next session: **Part 11 — System Architecture & Latency Analysis (Report Section 9)**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency + initial architecture, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison, **Part 6** statistical diagnostics, **Part 7** portfolio book + risk controls, **Part 8** Kalman time-varying hedge, **Part 9** nonlinear extension + RESET, **Part 10** multiple-testing correction + Diebold–Mariano. Now: **Part 11**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

> NOTE: the sandbox can be re-cloned between sessions (local branch may sit at the base commit with only README tracked). Recovery (used repeatedly): `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then `git reset --hard origin/arena/01a029cc-test-project`; then `bash environment/setup.sh`; `bash scripts/data/run_pipeline.sh`; `bash environment/check.sh` regenerates all git-ignored `results/`, `data/processed/`, `build/` outputs (~14 min). Re-export basket CSVs if `data/processed/engine/*.csv` missing: `python3 scripts/data/export_engine_input.py --target <T> --basket ...` for ADBE/AVGO/GS/CAT. If `scripts/model|statkit|analysis|plotting` or `report/` are missing, do this recovery, don't rewrite the parts.

## 2. Environment (re-verify each session)
`export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 52/52 PASS** (C++ unit tests **24 run / 0 failed, 2852 checks**; statkit **35/35** incl. new Diebold–Mariano `test_forecast.py`; kalman + nonlinear test modules pass; Parts 1-10 outputs regenerate). Pip is PEP-668 → `--user --break-system-packages`; **no TeX engine** (LaTeX canonical + HTML preview). Only pypi + GitHub reachable. Build `cmake -S src -B build && cmake --build build -j2`.

## 3. Part 10 delivered (verified 52/52, commit `9e198b3`)
- **`scripts/statkit/forecast.py`**: added `dm_return_pair(r1, r2, maxlags)` — Diebold–Mariano on a pair of *return* (net-P&L) series: mean of the daily difference divided by a Newey–West HAC standard error (correlated multi-day holds make an i.i.d. se invalid), two-sided p. (The existing `diebold_mariano` is the squared-loss forecast-error form and is left untouched.) Added `scripts/statkit/tests/test_forecast.py` (3 tests: forecast DM detects better forecast; return-pair detects a dominant strategy; equal strategies not rejected). statkit **32 → 35**.
- **`scripts/analysis/compare_multtest.py`**: builds the data-snooping universe as a **symmetric** grid: the same 11 hedge models {rolling OLS w60/w120/w180, Ridge w120 a10, Lasso w120, ElasticNet w120, PCA w120, Kalman q∈{1e-5,1e-4,1e-3,1e-2}} run on each of the four baskets = **M = 44** net-of-cost (10 bps) strategies on one common daily calendar (Nov 2010–Sep 2026, T=3979). Per strategy: net Sharpe, HAC t and one-sided p (mean net P&L > 0). Then Bonferroni over the 44 p-values; White (2000) Reality Check and Hansen (2005) SPA on the (T,44) net-return matrix (stationary bootstrap, 1,200 draws); pairwise Diebold–Mariano among the 11 ADBE hedges vs the best within-ADBE (Kalman q1e-5). Writes `results/tables/multtest_{audit,corrections,dm}.csv`.
- **`scripts/plotting/render_multtest.py`**: `multtest_sharpe.png` (top net Sharpe by basket), `multtest_pvalues.png` (log-scaled HAC p vs Bonferroni bar), `multtest_forest.png` (DM forest, 95% HAC CIs), + `report/tables/multtest.tex`. Wired into `report/build_report.py` (regeneration list + HTML embeds) and `environment/check.sh` Part-10 section incl. an **honesty gate** (n_survive_bonferroni must be 0 and White RC/SPA p>0.05) and an inline DM unit check; **44 → 52**.
- **Report §8**: replaced the "first multiple-testing statement" with the full **"A data-snooping correction of the headline Sharpe"** subsection (`\label{sec:diag:multtest}`) — hypothesis count, Bonferroni / White RC / Hansen SPA, explicit verdict, DM forest; updated the `sec:robustness` intro that previously deferred to Part 10.

### Part 10 honest result (net, 10 bps, common window Nov 2010–Sep 2026, M=44)
- **Empirical best of the scan: `GS` rolling OLS w180, full-sample net Sharpe ≈ 0.35** (90 round trips). Its raw one-sided HAC p is **0.059 — not even below the uncorrected 5% line** (sparse, autocorrelated P&L inflates any i.i.d. t; HAC tames it).
- **Bonferroni bar 0.05/44 ≈ 0.0011; best adjusted p = 1.0; 0 of 44 survive.** White RC p ≈ 0.87; Hansen SPA p ≈ 0.64. **No headline net Sharpe is distinguishable from zero after a fair correction.**
- **DM forest (ADBE):** no within-ADBE hedge pair is distinguishable — every 95% CI spans zero (e.g. Kalman q1e-5 vs OLS-w120 p≈0.68).
- Qualitatively defensible across Parts 4–9: mean reversion of the causal rolling spread; low-turnover smooth hedges (long window, low-q Kalman) net-positive vs churny refits; linear beats nonlinear net of cost; the within-ADBE Kalman q1e-5 is net-positive out-of-sample (test 2022+ net ≈ 0.11). Do **not** overclaim point estimates of net Sharpe in Part 11 or 12.

## 4. YOUR TASK — Part 11: System Architecture & Latency Analysis (Report Section 9)
PLAN Part 11: *full architecture writeup + latency/cache benchmarks in the report; vector-vs-deque and SoA-vs-AoS microbenchmarks; per-stage latency histograms (p50/p95/p99) plotted; design rationale (cache lines, prefetch, no-alloc, incremental Cholesky, CRTP); exit gate — all §6.1 latency/cache plots generated from real runs; §9 written; benchmark numbers traceable.*

**Important context before you start:** a large part of §9 was already written in **Part 3** and currently sits in `report/report.tex` under `\section{System Architecture and Latency Analysis}` (lines ≈1166–1274): Design principles and rationale, Cache-efficiency microbenchmarks (fig `cache_bench.png`), Per-stage latency (fig `latency_hist.png`, table `engine_bench.tex`). Do **not** rewrite what is already good and traceable. Your job is to **audit and complete** §9 so it is a tight, internally consistent, fully-live section that also reflects every C++ module actually shipped by Parts 4–10.

### 4.1 Suggested steps
1. **Audit §9 completeness against the C++ that actually exists.** `tree src/` (data, model, signal, portfolio, engine). Confirm the write-up covers the modules that grew *after* Part 3: the portfolio risk controls (`src/portfolio/risk.h` — inverse-vol weights, concentration caps, used by Part 7 and unit-tested), and any signal/spread components from Parts 4–9. The §9 principles list should mention the whole engine as it now stands; add a sentence where a shipped module is currently unmentioned.
2. **Verify every §9 number/figure is produced by a live script from a real run.** Check `scripts/analysis/capture_engine_bench.py` + `scripts/plotting/render_engine_results.py` produce `engine_bench.csv`, `latency_hist.png`, `cache_bench.png` and `report/tables/engine_bench.tex`; check they're wired into `report/build_report.py` regeneration and referenced by §9. If a figure/table §9 references is missing a live source, add it.
3. **Confirm the microbenchmarks are real runs and traceable.** vector-vs-deque and SoA-vs-AoS (cache_bench) and p50/p95/p99 latency histogram come from the C++ `--bench`/`--demo`/latency-hist code; the `--test` C++ suite (24 tests) is the reproducibility anchor. State measured numbers without inflation (Part 3 already does this well).
4. **Tie architecture claims to the earlier parts** (incremental Cholesky ↔ Parts 4–6 rolling hedges; low-latency book updates ↔ Part 7 risk; a note that the Part 8 Kalman recursion is a Python research reference whose port is deferred — §6 already has this note; keep §9 consistent).
5. **Only then write/refine the §9 narrative** and regenerate the HTML. This is primarily a *completeness + traceability* pass, not a rewrite.
6. **Tests + check.sh**: the C++ `--test` (24/0, 2852 checks) already guards the engine; add §9 traceability checks to `environment/check.sh` only if not already covered (engine_bench.csv + latency_hist.png + cache_bench.png + engine_bench.tex presence). `check.sh` must not regress below **52** and should grow only if you add genuinely useful checks.
7. **Report build**: `report/build_report.py` (LaTeX canonical; HTML preview in this sandbox). No TeX engine, so PDF compile is not verifiable here.

### 4.2 Exit gate (must pass before done)
- [ ] §9 System Architecture & Latency is complete and internally consistent, covering all shipped C++ modules (incl. post-Part-3 portfolio/risk).
- [ ] Every §9 figure/table (cache_bench, latency_hist p50/p95/p99, engine_bench) is produced by a live script from a real run and wired into the build; numbers traceable to source.
- [ ] C++ `--test` and all Python tests still pass; `environment/check.sh` ≥52 (grow only if useful).
- [ ] Report regenerated (HTML preview).
- [ ] `git status` clean after commit; `HANDOFF.md` rewritten for **Part 12** (per PLAN.md: Full Report Assembly + Non-Goals — complete Discussion/Conclusion, finalize all sections, verify build_report produces a complete consistent report).

### 4.3 Hints
- The honest bar still applies: report §9 numbers as measured, with the caveats Part 3 already gives (single target/basket timings; block-deque/AoS prefetch fine at these sizes → modest microbenchmark multipliers). Don't invent a big latency win the machine doesn't show.
- Keep `results/` git-ignored; commit only sources + tracked report artifacts.
- If you find §9 already fully traceable, Part 11 is a *small* pass — that is fine and correct; say so and move the HANDOFF to Part 12. Do not pad.
