# HANDOFF — Next session: **Part 7 — Portfolio-Level Construction + Risk Controls**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison (OLS/Ridge/Lasso/ElasticNet/PCR), **Part 6** statistical diagnostics on the final model(s). Now: **Part 7**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

> NOTE: the sandbox workspace was re-cloned once at the top of the Part-6 session (local branch pointed at the base commit `541d389`). All prior work was intact on the remote branch. Recovery used `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then `git reset --hard origin/arena/01a029cc-test-project`, followed by re-running `environment/setup.sh`, `scripts/data/run_pipeline.sh`, `cmake`, and the result-generating scripts (everything under `results/`, `data/processed/`, `build/` is git-ignored and must be regenerated after any fresh clone). If the local tree is ever at `541d389` with no `scripts/model`, `scripts/statkit`, etc., re-do that recovery instead of re-writing the parts.

## 2. Environment (re-verify each session)
Run `export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 28/28 PASS.** If the sandbox reset packages/data, run in order: `bash environment/setup.sh` → `bash scripts/data/run_pipeline.sh` → `cmake -S src -B build && cmake --build build -j2` → then `bash environment/check.sh` (which regenerates all results via `report/build_report.py`).
- pip PEP-668 → `pip install --user --break-system-packages`. No TeX engine (report = canonical LaTeX + HTML preview). Only pypi + GitHub reachable.
- `data/processed/universe.csv` = 195 tickers (yfinance), QA 10/10, median 4190 days; `data/processed/engine/ADBE.csv` (target ADBE; CRM/ADSK/INTU) exported by `scripts/data/export_engine_input.py`.
- Build: `cmake -S src -B build && cmake --build build -j2`. Binaries `build/statarbsim` (--version/--test/--demo/--backtest), `build/microbench`.

## 3. Part 6 delivered (verified 28/28; statkit 29/29; C++ 22 tests / 2837 checks)
- **`scripts/analysis/run_diagnostics.py`** runs the full spec §2.2 battery on the ADBE-vs-basket residuals using the existing `scripts/statkit/` (ADF/KPSS via `interpret_stationarity`, EG + Johansen, Durbin-Watson + Ljung-Box, Breusch-Pagan + White, Newey-West HAC via `hac.fit_and_compare_se`, OU half-life, first Bonferroni multiple-testing bound). Compares a **static** full-sample OLS/Ridge hedge with the **causal rolling** spread the strategy trades, over full sample and 2022+ test. Writes `results/tables/{diagnostics_summary,diagnostics_halflife,diagnostics_coint,diagnostics_multtest,spread_residuals}.csv`.
- **`scripts/plotting/render_diagnostics.py`** → `results/figures/diagnostics_{residual,halflife}.png` + `report/tables/diagnostics.tex` (2 tables: `tab:diag_stationarity`, `tab:diag_coint`). Both wired into `report/build_report.py` (regeneration list + HTML preview embeds figures + a diagnostics battery table); `environment/check.sh` gains 4 Part-6 checks.
- **Report §8** (Robustness and Statistical Validity) written with subsections: results (D1-D3 findings), cointegration & hedge significance, half-life vs holding, first multiple-testing statement; Part 10 still pending (formal Bonferroni/White/Hansen SPA + DM table).
- **Key honest numbers** (target ADBE vs {CRM,ADSK,INTU}, log prices): static full-sample OLS/Ridge hedge residual → ADF p≈0.08-0.09 (NOT reject unit root), OU half-life ~170-178 bars (test ~460-475) → verdict *unit_root*: the fixed long-run hedge is NOT a clean cointegrating spread. The **causal rolling spread** (what's traded) → ADF p<0.001, verdict *stationary* in 2022+ (full = ambiguous), **OU half-life ≈9.4 bars stable** across full/test. EG pair ADF p: CRM 0.003, ADSK 0.033, INTU 0.117; **Johansen rank r=2** on the 4-name vector. HAC t on full-sample hedge: CRM 29.5→7.3, ADSK 29.7→9.4, INTU 1.5→0.5 (INTU insignificant). DW<0.15, Ljung-Box p<10^-3, BP/White p<10^-3 (strong serial corr + heteroskedasticity → HAC needed). Realized holding ~16-18 bars = **~1.7-1.9 half-lives** per trade. First multiple-testing: M=13 grid candidates scanned → Bonferroni bar p<0.0038; none of the net Sharpes approaches it, honest preliminary verdict = real-in-sign but not strong enough after correction (Part 10 formalizes).

## 4. YOUR TASK — Part 7: Portfolio-Level Construction + Risk Controls
Make it a **book**, not a single strategy (PLAN Part 7).

### 4.1 Deliverables (suggested)
1. **Multiple concurrent basket trades.** Choose a set of (target, basket) relationships (e.g. several software/tech names against their peers, or a handful across sectors) and run them through the causal cost-aware engine (Python `scripts/model/baseline_models.py` + `simulate`, or the C++ engine per basket) over the common panel. Combine the per-basket P&L streams into one portfolio NAV.
2. **Risk controls, enforced and verified**: inverse-vol / risk-parity sizing across active baskets; VaR / CVaR on the daily portfolio P&L; gross/net exposure limits; optional sector-neutrality. Verify in the backtest that the limits are never breached (assertions), not just reported.
3. **Regime segmentation + stress periods**: split the daily portfolio P&L by volatility regime (e.g. VIX terciles if available; otherwise realized vol terciles) and report performance (Sharpe, maxDD, VaR) conditional on regime; explicitly report stress windows present in the data (COVID-2020 and 2022 are inside the sample; 2008 is not — say so honestly).
4. **Sensitivity heatmaps**: lookback (window) × Ridge λ (or z-thresholds) × net Sharpe (or net-VaR) heatmaps for the chosen models; generated as figures.
5. **Tests + check.sh**: extend C++ `--test` (e.g. inverse-vol sizing, gross/net limit enforcement) and/or statkit tests if you add Python risk routines. `environment/check.sh` total must not regress below 28 and should grow.
6. **Report**: write the portfolio section prose + live figures/tables into `results/` + `report/tables` + `report/figures`, wire into `report/build_report.py`, regenerate HTML. Numbers from the run.

### 4.2 Exit gate (must pass before done)
- [ ] Portfolio risk controls (sizing, VaR/CVaR, gross/net limits) are enforced and verified in the multi-basket backtest.
- [ ] Regime/stress table produced (with honest stress-window coverage statement).
- [ ] Sensitivity heatmaps (lookback × λ, z-threshold) generated.
- [ ] New tests pass; all prior tests still pass; `environment/check.sh` ≥28 and grows if practical.
- [ ] Report section regenerated (HTML preview; LaTeX canonical); `git status` clean after commit; `HANDOFF.md` written for **Part 8** (Kalman-filtered time-varying beta).

### 4.3 Hints
- Multi-basket: export a handful of `data/processed/engine/<TARGET>.csv` wide matrices via `export_engine_input.py` with different `--basket` choices; the engine/`simulate` path is per-basket, so aggregate P&L at the daily level and then impose book-level risk controls on the aggregated stream.
- VaR/CVaR can be parametric or historical on the daily P&L; report both and note which. Regime conditioning uses whatever vol proxy is honest given only OHLCV (realized vol terciles over the portfolio; if a VIX series is not available in-sandbox, disclose and use realized vol).
- The daily edge is small and cost-sensitive (Parts 4-5); a portfolio may or may not be net-positive. Report the honest multi-basket result; the value of Part 7 is the risk-engineering (controls enforced + verified), not a manufactured positive Sharpe.
- Keep results/ git-ignored; commit only sources + tracked report artifacts meant to be tracked.
