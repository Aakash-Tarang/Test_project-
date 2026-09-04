# HANDOFF — Next session: **Part 9 — Nonlinear Extension + RESET Test**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison (OLS/Ridge/Lasso/ElasticNet/PCR), **Part 6** statistical diagnostics on final models, **Part 7** portfolio-level construction + risk controls, **Part 8** Kalman-filtered time-varying beta. Now: **Part 9**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

> NOTE: the sandbox can be re-cloned between sessions (local branch may sit at the base commit with only README tracked). Recovery (documented and used twice already): `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then `git reset --hard origin/arena/01a029cc-test-project`; then `bash environment/setup.sh`; `bash scripts/data/run_pipeline.sh`; `cmake -S src -B build && cmake --build build -j2`; then `bash environment/check.sh` regenerates all git-ignored `results/`, `data/processed/`, `build/` outputs (~6-7 min). Also re-export basket CSVs: `python3 scripts/data/export_engine_input.py --target <T> --basket ...` for ADBE/AVGO/GS/CAT if `data/processed/engine/*.csv` are missing. If `scripts/model`/`scripts/statkit`/`results/figures/portfolio_*.png` are missing, do this recovery, don't rewrite the parts.

## 2. Environment (re-verify each session)
`export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 36/36 PASS** (C++ unit tests **24 run / 0 failed, 2852 checks**; statkit 29/29; kalman python test passes; backtest reproduces Part-4 numbers; Parts 4-8 outputs regenerate). Pip is PEP-668 → `--user --break-system-packages`; no TeX engine (LaTeX canonical + HTML preview). Only pypi + GitHub reachable. Build `cmake -S src -B build && cmake --build build -j2`.

## 3. Part 8 delivered (verified 36/36)
- **`scripts/model/kalman.py`** (new): dependency-free Kalman-filtered time-varying hedge. State = (intercept, k betas); random-walk drift β_t=β_{t-1}+η (Q=diag q²R), obs y_t=[1,x_t]β_t+ε (R from warm-up OLS residual variance). Standard predict/update; **predictive state** β_{t|t-1} scores bar t (causal, one-bar delay, mirrors rolling OLS timing), z_t = standardized innovation; bar t fed in after scoring. Output API-compatible with baseline_models.walk_forward (z, beta, intercept) so it drops into simulate().
- **`scripts/test_kalman.py`** (new, python): synthetic unit checks — filtered β tracks a drifting true β (RMSE 0.73 vs 3.37 for fixed OLS), predictive beta at bar t uses only bars <t (causal), filtered residual stationary (ADF p<10^-3, OU half-life ~8 bars). All pass.
- **`scripts/analysis/compare_kalman.py`**: on ADBE/{CRM,ADSK,INTU}, sweeps Kalman state-noise q={1e-5..1e-1} and compares (full 2010-2026 + test 2022+) to rolling OLS w60/120/180 and rolling Ridge w120 a=10, all net at 10 bps; q* chosen on 2019-2021 validation by net Sharpe (→1e-3, disclosed as fragile). Writes `results/tables/kalman_summary.csv` + `results/model/beta_kalman.csv`. `scripts/plotting/render_kalman.py` → `results/figures/kalman_{sharpe,beta}.png` + `report/tables/kalman.tex`. Wired into `report/build_report.py` (regeneration + HTML embeds 2 figs + summary table) and `environment/check.sh` gains 4 checks (total 32→36). Report §6 gets a full "Kalman-filtered time-varying hedge" subsection (`\label{sec:kalman}`) with findings K1-K3.
- **Key honest numbers** (net, 10 bps; ADBE basket): best Kalman is at **LOW state noise** q=1e-5 → full net Sharpe **0.113** and test 2022+ net **0.111** (both positive; only 61 round trips), beating rolling Ridge w120 (0.093 / test -0.031) and rolling OLS w120 (0.050 / test -0.302). High q (fast adaptation, best fit likelihood) **over-tracks and destroys the spread** (net -0.04 to -0.93). Only very-short rolling OLS w60 also positive on test (0.518). Honest caveat: single q is fragile; validation-selected q=1e-3 does poorly OOS → "how much to adapt" is regime-dependent and should lean toward the smooth/low-turnover limit given the small cost-drag-limited edge.

## 4. YOUR TASK — Part 9: Nonlinear Extension + RESET Test
Honestly evaluate whether *nonlinear* signal extraction adds value over the linear hedges, and run the Ramsey RESET misspecification test on the linear model. PLAN Part 9; report §6 still has nonlinear text + C++ port note pending.

### 4.1 Deliverables (suggested)
1. **Nonlinear predictors** on the ADBE basket (or a small set of baskets): gradient-boosted trees (sklearn) and/or a shallow MLP, fed with causal features (lags of the spread residual, basket/target returns, realized vol) to predict next-period spread change or the reversion; compare out-of-sample to the linear baselines (rolling OLS/Ridge/Kalman) on the SAME no-lookahead, cost-aware net metrics. Watch for lookahead in feature construction.
2. **Ramsey RESET test** (spec §2.2 / linearity misspecification) on the base linear model: regress residual on fitted^2, fitted^3 powers; H0: no omitted nonlinearity. Run with statkit or statsmodels; report honestly (a RESET rejection would justify the nonlinear extension; failure to reject would mean it likely adds nothing).
3. **Learning curves** (train-size vs OOS error) for the chosen nonlinear model to show bias-variance; honest bias-variance discussion.
4. **Verifiable no-lookahead**: nonlinear models can overfit; enforce strictly causal train/test split (no future leakage), report both gross and net (10 bps).
5. **Tests + check.sh**: unit checks for the nonlinear pipeline if you add a module (Python is fine) and for RESET. `--test`/check.sh must not regress below 36 and should grow.
6. **Report**: write the nonlinear subsection in §6 + live figures/tables into `results/` + `report/tables` + `report/figures`, wire into `report/build_report.py`, regenerate HTML. State clearly whether nonlinearity beats linear after costs (Part 9's honest verdict).

### 4.2 Exit gate (must pass before done)
- [ ] Nonlinear vs linear OOS R²/Sharpe compared with an honest conclusion (in-sample gains rarely survive OOS + costs).
- [ ] Learning curves + RESET test plotted and reported.
- [ ] New unit tests pass; all prior tests still pass; `environment/check.sh` ≥36 and grows if practical.
- [ ] Report section regenerated (HTML preview; LaTeX canonical).
- [ ] `git status` clean after commit; `HANDOFF.md` written for **Part 10** (per PLAN.md).

### 4.3 Hints
- The honest bar is high and Parts 4-8 already show the daily edge is small and cost-limited; the likely truthful conclusion is that nonlinearity adds little or nothing net-of-cost on this basket. Report that if it's true.
- RESET + a shallow GBM/MLP comparison plus learning curves is enough; you don't need to re-run the full 195-name universe.
- sklearn available (1.9.0); keep the report consistent with §6. Keep results/ git-ignored; commit only sources + tracked report artifacts.
