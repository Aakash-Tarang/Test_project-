# HANDOFF — Next session: **Part 10 — Multiple-Testing Correction + Diebold–Mariano + Final Statistical Validity**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison (OLS/Ridge/Lasso/ElasticNet/PCR), **Part 6** statistical diagnostics on final models, **Part 7** portfolio-level construction + risk controls, **Part 8** Kalman-filtered time-varying beta, **Part 9** nonlinear signal extraction (GBM/MLP) + Ramsey RESET. Now: **Part 10**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

> NOTE: the sandbox can be re-cloned between sessions (local branch may sit at the base commit with only README tracked). Recovery (documented and used repeatedly): `git fetch origin '+refs/heads/*:refs/remotes/origin/*'` then `git reset --hard origin/arena/01a029cc-test-project`; then `bash environment/setup.sh`; `bash scripts/data/run_pipeline.sh`; `bash environment/check.sh` regenerates all git-ignored `results/`, `data/processed/`, `build/` outputs (now ~12 min, dominated by the Part 9 expanding-window drivers). Also re-export basket CSVs: `python3 scripts/data/export_engine_input.py --target <T> --basket ...` for ADBE/AVGO/GS/CAT if `data/processed/engine/*.csv` are missing. If `scripts/model`/`scripts/statkit`/`results/figures/portfolio_*.png` are missing, do this recovery, don't rewrite the parts.

## 2. Environment (re-verify each session)
`export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 44/44 PASS** (C++ unit tests **24 run / 0 failed, 2852 checks**; statkit 32/32; kalman python test passes; nonlinear pipeline tests pass; Parts 1-9 outputs regenerate). Pip is PEP-668 → `--user --break-system-packages`; no TeX engine (LaTeX canonical + HTML preview). Only pypi + GitHub reachable. Build `cmake -S src -B build && cmake --build build -j2`.

## 3. Part 9 delivered (verified 44/44, commit `d7c5b79`)
- **`scripts/statkit/misspec.py`** (new): Ramsey RESET `ramsey_reset(y, X, powers=(2,3), test='f'/'lm')` — restricted linear fit, add centered powers of fitted values, F or LM joint test; returns dict incl. `statistic/pvalue/reject_linear_at_5pct`. Registered in `scripts/statkit/__init__.py`. `scripts/statkit/tests/test_misspec.py` (3 tests: accepts linear, rejects quadratic, F/LM decisions agree). statkit suite **29 → 32**.
- **`scripts/model/nonlinear_models.py`** (new): causal forecasters. `causal_spread` (rolling-OLS-120 hedge), `build_features_targets(lprices, window=120, h=5)` → 11 causal features + realized short-side target `R_t^h = -(S_{t+h}-S_t)/g_t`; `walk_forecast(lprices, model='linear'|'gbm'|'mlp', window=120, h=5, retrain_every=63)` = expanding-window refit every 63 bars, train-fold scaler applied to held-out bars (no lookahead). (The old dead `walk_forecast`/`_predict_standardized` draft leak was removed; `_fit_predict` is gone; single clean `walk_forecast` is the API.)
- **`scripts/test_nonlinear.py`** (new): 4 unit tests (feature/target finiteness+bounded, walker runs all estimators with many held-out preds, no-lookahead correlation < 0.95, RESET accepts linear & rejects quadratic).
- **`scripts/analysis/compare_nonlinear.py`**: per-basket OOS (2022+) IC/R² for linear/GBM/(MLP on ADBE only), Ramsey RESET on the linear forecast eqn (training rows), net-of-cost gated trading vs Part-4 baseline, GBM learning curve. Writes `results/tables/nonlinear_{summary,reset,trading,learning}.csv`. `scripts/plotting/render_nonlinear.py` → 4 figures (`nonlinear_ic/reset/sharpe/learning.png`) + `report/tables/nonlinear.tex`. Wired into `report/build_report.py` (regeneration + HTML embeds). `environment/check.sh` gains a Part-9 section incl. an **honesty gate** (nonlinear gate must NOT beat baseline net Sharpe full-sample); **36 → 44**.
- **Report §6** new subsection **"Nonlinear signal extraction (GBM, MLP, RESET)"** (`\label{sec:nonlinear_sig}`) with findings N1-N3 + net-of-cost verdict.

### Part 9 honest result (net, 10 bps, ADBE basket unless noted)
- **Linear wins the forecasting race**: OOS (2022+) IC on `R_t^h`: ADBE linear 0.34 / gbm 0.26 / mlp 0.27; AVGO 0.42/0.37; GS 0.30/0.22; CAT 0.24/0.17. Linear OOS R² 4-17%; GBM negative OOS R² in most baskets.
- **RESET mixed**: rejects linearity at 5% on AVGO (p≈0), GS (0.002), CAT (0) but **not on ADBE (p≈0.38)** — the basket used throughout Parts 4-8.
- **Learning curve**: GBM OOS R² (2019-20 val) ≈ -0.45 at 400 train bars, only turns positive near ~2,800 bars.
- **Net-of-cost verdict is negative**: gating the Part-4 z-threshold book (ADBE) so entries require forecaster reversion confirmation never raises net Sharpe — full-sample baseline 0.050 vs linear-gate 0.037, gbm-gate 0.003, mlp-gate -0.018 (test 2022+ all ≈ -0.30 to -0.35). **Conclusion (reported): nonlinearity adds nothing reliable net of cost**; the linear base model was near the practical limit. Do not overclaim in Part 10 or later.

## 4. YOUR TASK — Part 10: Multiple-Testing Correction + Diebold–Mariano + Final Statistical Validity
The adversarial self-critique: after scanning many baskets/windows/models across Parts 4-9, how much of the "best" Sharpe survives a full multiple-testing correction, and is any model pair's difference statistically distinguishable? Statkit already ships `bonferroni`, `white_reality_check`, `hansen_spa`, `bootstrap_ci`, `stationary_bootstrap` (Part 2, spec §2.3). You will ADD a **Diebold–Mariano** test (not yet in statkit) and assemble the full corrected numbers.

### 4.1 Deliverables (suggested)
1. **Full hypothesis-count audit.** Count every *tested* choice that could inflate the reported Sharpe: baskets scanned (4 in the real data study; also the wider cross-sections used in Parts 1-4/validation), hedge estimators (OLS w60/120/180, Ridge, Lasso, ElasticNet, PCR, Kalman q-sweep, linear/GBM/MLP signal gates), window/entry/exit choices, and the Part-9 honesty regime. Tally M = total number of effectively-compared strategies.
2. **Multiple-testing corrections** over the set of model/basket net-Sharpe statistics: **Bonferroni** (report both corrected significance threshold and which, if any, survive), **White (2000) Reality Check** and **Hansen (2005) SPA** via the existing statkit functions (block bootstrap), applied to the best model's realized performance.
3. **Diebold–Mariano test** across model pairs (e.g. rolling OLS-120 vs Ridge vs best-Kalman; linear vs GBM/MLP forecast-gated books) on the net-return difference series with HAC variance; produce a **forest plot** of DM statistics / mean difference with CIs across pairs. Add the DM implementation to statkit with unit tests (synthetic: correctly detects equal vs. better model).
4. **Explicit numeric answer** to the spec question: **"is the best Sharpe still distinguishable from zero after correction?"** Report the corrected p-value(s) honestly (Part 8/9 already hint the daily edge is small and cost-limited — the corrected best Sharpe may or may not clear a FWER/FDR bar; give the real number either way).
5. **Tests + check.sh**: unit tests for DM (+ any new helper). `--test`/check.sh must not regress below **44** and should grow.
6. **Report**: complete **§8 Robustness / Statistical Validity** — it currently has "A first multiple-testing statement" (near end of §8); replace/extend it with the full corrected audit + DM forest plot, wired live into `report/build_report.py` + HTML.

### 4.2 Exit gate (must pass before done)
- [ ] Explicit numeric answer to "is the best Sharpe still distinguishable from zero after correction?" (with the corrected method and p-value).
- [ ] DM results table across model pairs + forest plot present and reported.
- [ ] Hypothesis-count audit (M) stated explicitly.
- [ ] New unit tests pass; all prior tests still pass; `environment/check.sh` ≥44 and grows if practical.
- [ ] Report §8 regenerated (HTML preview; LaTeX canonical).
- [ ] `git status` clean after commit; `HANDOFF.md` written for **Part 11** (per PLAN.md: System Architecture & Latency Analysis, Report Section 9).

### 4.3 Hints
- **Honesty is the deliverable.** Parts 4-9 established the edge is small, cost-limited, and that over-adaptation/nonlinearity do not reliably help. Part 10's job is to answer squarely whether anything survives a real correction — the likely truthful answer is that the *single best pre-selected* Sharpe is marginal-to-not-significant after a strict FWER over the full search, while the low-turnover Kalman/hedge-level finding and the out-of-sample (2022+) positivity on the short window remain the most defensible qualitative claims. Say exactly that with the numbers.
- DM needs HAC variance of the loss/return difference (statkit `hac.py` already exists). Correlated daily obs → do NOT use iid variance.
- Block bootstrap (stationary_bootstrap in `bootstrap.py`) is the natural engine for White RC / SPA; keep seed fixed for reproducibility.
- Reuse the *net-of-cost* return series the simulators already emit (net Sharpe is the defensible object; report gross too but correct the net one). Keep report consistent with §8. Keep `results/` git-ignored; commit only sources + tracked report artifacts.
