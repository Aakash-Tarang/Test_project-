# HANDOFF — Next session: **Part 8 — Kalman-Filtered Time-Varying Beta**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison (OLS/Ridge/Lasso/ElasticNet/PCR), **Part 6** statistical diagnostics on the final model(s), **Part 7** portfolio-level construction + risk controls. Now: **Part 8**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

> NOTE: the sandbox may be re-cloned between sessions (local branch can appear at base commit `541d389` while all prior work is intact on the remote branch). Recovery: `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then `git reset --hard origin/arena/01a029cc-test-project`; then `bash environment/setup.sh`; `bash scripts/data/run_pipeline.sh`; `cmake -S src -B build && cmake --build build -j2`. Then `bash environment/check.sh` regenerates all git-ignored `results/`, `data/processed/`, `build/` outputs (~5 min). If `scripts/model`/`scripts/statkit`/`results/figures/portfolio_*.png` are missing, do this recovery, don't re-write the parts.

## 2. Environment (re-verify each session)
Run `export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 32/32 PASS** (C++ unit tests **24 run / 0 failed, 2852 checks**; statkit 29/29; backtest reproduces Part-4 numbers; all 6 Parts' outputs regenerate). Pip is PEP-668 → `--user --break-system-packages`; no TeX engine (LaTeX canonical + HTML preview). Only pypi + GitHub reachable. Build `cmake -S src -B build && cmake --build build -j2`.

## 3. Part 7 delivered (verified 32/32)
- **`src/portfolio/risk.h`** (new, dependency-free): inverse-vol weights, gross/net exposure, `within_limits` (gross cap, net cap, concentration cap). Unit-tested: inverse-vol ordering/normalization/zero-vol handling; gross/net/limit breach detection. C++ tests grew 22→24.
- **`scripts/analysis/capture_portfolio.py`**: builds a 4-strategy market-neutral *book* (each OLS 120-bar causal hedge, 10 bps one-way): ADBE vs {CRM,ADSK,INTU}, AVGO vs {ADI,INTC,AAPL}, GS vs BAC, CAT vs HON (disjoint names, 3 sectors; engine CSVs exported to `data/processed/engine/`). Aligns daily P&L on a common calendar, applies **inverse-vol allocation (monthly rebalance, per-name water-fill cap 45%, sum=1)** vs equal weight, **historical VaR95/CVaR95**, **regime (realized-vol terciles) + stress (COVID-2020, 2022)** table, and two **net-Sharpe heatmaps** (window×entry-z and window×ridge-λ on the ADBE book). Constraint flags (`sum(w)=1`, `max w ≤ cap`) asserted on every rebalance. Writes `results/tables/portfolio_{summary,daily,regime,heat_ez,heat_ridge}.csv`.
- **`scripts/plotting/render_portfolio.py`**: → `results/figures/portfolio_{equity,allocation,risk,heat_ez,heat_ridge}.png` + `report/tables/portfolio.tex` (2 tables: summary + regime). Wired into `report/build_report.py` (regeneration list + HTML preview embeds 5 figures + a portfolio summary table); `environment/check.sh` gains 4 Part-7 checks (total 28→32). Report §7 gains subsection **"Multi-basket book and risk controls"** (`\label{sec:portfolio}`) with figures/tables and the honest-outcome prose.
- **Key honest numbers** (net, 10 bps): solo books ADBE Sharpe 0.05/ann 0.004/maxDD 0.42; AVGO -0.42; GS -0.08; CAT -0.44. Combined book **net-negative** (honest, consistent with Parts 4-5): equal-wt Sharpe -0.46/ann -0.020/maxDD 0.44/VaR95 0.00409; **inverse-vol improves tail**: Sharpe -0.40, maxDD 0.38, VaR95 0.00389, CVaR95 0.00628. Regime/stress worst in **2022 selloff (net Sharpe -1.31)** and mid/high vol terciles; COVID-2020 -0.21. 2008 NOT in sample (disclosed). Heatmaps: higher entry-z + longer window raise net Sharpe (turnover/cost reduction), e.g. window180/entry3.0 ≈0.21 vs default ≈0.05; strong ridge helps short windows.

## 4. YOUR TASK — Part 8: Kalman-Filtered Time-Varying Beta
Make the hedge *genuinely adaptive* (state-space) and compare it to the fixed-window rolling OLS on the same no-lookahead, cost-aware pipeline. PLAN Part 8; report §5/§6 placeholder for Kalman still pending.

### 4.1 Deliverables (suggested)
1. **Kalman filter for the time-varying hedge** on target ADBE vs {CRM,ADSK,INTU}: state-space `β_t = β_{t-1} + η_t` (random-walk drift, η ~ N(0,Q)), observation `y_t = X_t β_t + ε_t` (ε ~ N(0,R)); standard scalar- or vector-Kalman recursion (predict: β^-_t, P^-_t; update: Kalman gain K_t, β̂_t, P_t). In log-price space like the rest. Implement in Python (reference) and, if practical, mirror the recursion in a small dependency-free C++ header + unit test (state-space update). Time the per-bar step with `LatencyHist`.
2. **Adaptive state/vol estimates**: initial β via warm-up OLS; Q (state noise) as a fraction of the innovation variance or a small multiple tuned on a validation split; R from residual variance. Document how Q/R are set (hyper-parameters), chosen no-lookahead on validation.
3. **No-lookahead causal z + backtest**: from the filtered β̂_t (causal, uses data through t) compute the spread residual z_t and run it through the Part-4/5 `simulate` at 10 bps to get net Sharpe. Compare to rolling OLS (window 60/120/180) on the same full-sample and 2022+ test spans (reuse Part 5 style).
4. **Diagnostics tie-in**: does the Kalman spread show the stationary, ~9-bar half-life property of Part 6? Report OU half-life and ADF/KPSS on the Kalman residual (statkit) so the adaptive comparison is on the same footing.
5. **Tests + check.sh**: add a Kalman unit test (tracking error: filtered β tracks a slowly-varying true β; covariance shrinkage vs rolling OLS). `--test` grows above 24; `environment/check.sh` must not regress below 32 and should grow.
6. **Report**: write the Kalman subsection (in §5 Model Comparison and/or a §6 Nonlinear/Adaptive extension) + live figures/tables into `results/` + `report/tables` + `report/figures`; wire into `report/build_report.py`; regenerate HTML. Honest verdict vs rolling OLS.

### 4.2 Exit gate (must pass before done)
- [ ] Kalman (time-varying beta) implemented and compared no-lookahead to rolling OLS; tracking-error + OOS comparison shown.
- [ ] Q/R (state/observation noise) hyper-parameters documented and chosen on validation.
- [ ] New unit test(s) pass; all prior tests still pass; `environment/check.sh` ≥32 and grows if practical.
- [ ] Report section written and regenerated (HTML preview; LaTeX canonical).
- [ ] `git status` clean after commit; `HANDOFF.md` written for **Part 9** (Nonlinear Extension + RESET test).

### 4.3 Hints
- State dimension = n_features (+ optional intercept). Use log prices as in Parts 4-6. Compare the filtered β̂ path (Figure) vs the rolling-OLS β from `results/model/beta_ols.csv`.
- The honest bar is high: rolling OLS already gives the stationary ~9-bar spread. Kalman should show whether continuous adaptation (no window) improves tracking/keeps the spread stationary with fewer parameters or lower turnover — report whichever is true (it may not beat rolling OLS net of costs).
- Keep results/ git-ignored; commit only sources + tracked report artifacts.
