# HANDOFF — Next session: **Part 6 — Statistical Diagnostics on the Final Model(s)**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest), **Part 5** baseline model comparison (OLS/Ridge/Lasso/ElasticNet/PCR). Now: **Part 6**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

## 2. Environment (re-verify each session)
Run `export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 24/24 PASS.** If packages/tools are missing at session start, run `bash environment/setup.sh` first.
- pip PEP-668 → `pip install --user --break-system-packages`. No TeX engine (report = canonical LaTeX + HTML preview). Only pypi + GitHub reachable.
- Data persists (`data/processed/universe.csv`, 195 tickers, QA 10/10). If missing: `python3 scripts/data/clean.py --source yfinance && python3 scripts/data/qa.py`.
- Build: `cmake -S src -B build && cmake --build build -j2`. Binaries `build/statarbsim` (--version/--test/--demo/--backtest), `build/microbench`.

## 3. Part 5 delivered (verified 24/24)
- **`src/model/basket_selector.h`** completed: all 5 baseline estimators, dependency-free, with a free intercept (centering internally) returning per-feature slopes + intercept in original scale. OLS/Ridge via centered normal equations (Cholesky); Lasso/ElasticNet via coordinate descent on unit-norm standardized design (objective `½||y−U b||²+αΣ_j[ρ|b_j|+(1−ρ)/2 b_j²]`, soft-threshold update, mapped back to raw scale); PCA = principal-component regression via cyclic-Jacobi eig of the centered covariance (keep top_k). Public API `select(X,n_obs,n_feat,y,BasketParams{alpha,l1_ratio,top_k,tol,max_iter})`.
- **Unit tests** added (6): OLS recovers β, Ridge→OLS as λ→0 + shrinks, Lasso sparsifies (heavier penalty ⇒ sparser), ElasticNet between + deterministic, PCR(k=n)=OLS, PCR R² rises with k. **`--test`: 22 run / 0 failed; checks 2837.** All prior tests still pass.
- **Real-data comparison** (new): `scripts/model/baseline_models.py` (estimators + causal `walk_forward` producing z,β-intercepts + `simulate` replicating Part 4 one-bar-delay, cost-aware logic + `information_coefficient`) and `scripts/analysis/compare_models.py` (runs the grid, tunes hyper-params on a 2019-2021 validation split by net Sharpe, reports out-of-sample 2022+ test metrics + full-sample context → `results/tables/model_compare_summary.csv`, `model_compare_sweep.csv`, `results/model/beta_<m>.csv`, `net_test_<m>.csv`). `scripts/plotting/render_model_comparison.py` → `results/figures/model_compare_{sharpe,ic,stability}.png` + `report/tables/model_compare.tex`, all wired into `report/build_report.py` and embedded in the HTML preview; check.sh gains 3 Part-5 checks. `results/model/*` added to `.gitignore`.
- **Honest headline numbers** (10 bps cost, window 120, entry/exit 2.0/0.5): Python OLS full-sample net Sharpe ≈0.05 exactly reproduces the Part 4 C++ engine (harness validated). Full-sample net Sharpe by estimator: OLS 0.050, **Ridge(α=10) 0.094** (best), Lasso 0.007, ElasticNet −0.032, PCA(k=1) 0.026. Out-of-sample (2022+) IC is positive for all (0.12–0.21; PCA highest) but all net-negative after costs in that adverse regime. Report §5 (Model Comparison Results) fully written with methodology, findings (M1–M5), figures and table.

## 4. YOUR TASK — Part 6: Statistical Diagnostics on the Final Model(s)
Run every diagnostic in spec §2.2 on the residuals of the final model(s) and report honestly (this is where a real "does the spread mean-revert / is the hedge real" verdict is made). statkit already implements the battery and is unit-tested (`scripts/statkit/`, run via `scripts/run_statkit_tests.py`).

### 4.1 Deliverables (suggested)
1. **Pick the final residual(s).** From Part 5 the natural candidates are the Ridge hedge residual (best full-sample net) and the OLS spread (the Part 4 baseline); PCA if you want its high-IC angle. Use the causal, walk-forward residuals (already produced) — do NOT refit in-sample for diagnostics.
2. **Run the §2.2 battery** on the chosen residual series and tabulate: ADF & KPSS (stationarity), Engle–Granger (and/or Johansen at basket level) cointegration, Durbin–Watson & Ljung–Box autocorrelation, Breusch–Pagan & White heteroskedasticity, Newey–West HAC t-stats (show before/after on the hedge betas), and an OU half-life fit (§4 diagnostics text in report already derives all of these; statkit has the routines). Report honestly — a residual that fails ADF (unit root) means the hedge is spurious and must be flagged, not buried.
3. **Half-life vs holding-period mismatch.** Compare the OU half-life τ_{1/2} to the realized average holding from Part 4/5 (~16–30 bars). If holding ≫ few half-lives (or ≪), say so explicitly and discuss what it implies.
4. **Multiple-testing note.** Give a first numeric multiple-testing answer on the models already scanned (Part 10 will formalize with Bonferroni/White/Hansen), e.g. how many hypotheses the Part 5 comparison + grid entail and what the corrected bar on "best Sharpe different from zero" is.
5. **Tests + check.sh.** Extend `--test` if you add diagnostics to the C++ side (statkit is Python). `environment/check.sh` total must not regress below 24 and should grow.
6. **Report.** Fill/refresh the relevant subsection(s) — the methodology text (§4 diagnostics subsections) is written; produce a live results table + any figures into `results/` + `report/tables` + `report/figures`, wire into `report/build_report.py`, regenerate HTML. Numbers must come from the run.

### 4.2 Exit gate (must pass before done)
- [ ] All §2.2 diagnostics run and are tabulated for the final model(s), with an honest verdict on each (stationarity, cointegration, autocorrelation, heteroskedasticity, HAC before/after).
- [ ] Half-life-vs-holding-period mismatch explicitly discussed with numbers.
- [ ] A first numeric multiple-testing statement for the models scanned so far.
- [ ] Prior tests still pass; `environment/check.sh` ≥24 and grows if practical.
- [ ] Report section regenerated (HTML preview; LaTeX canonical); `git status` clean after commit; `HANDOFF.md` written for **Part 7** (per PLAN.md).

### 4.3 Hints
- The diagnostics belong in the appendix/live-tables of the report but the verdict belongs in § (Robustness/validity or a "diagnostics" subsection). Read how statkit APIs are exposed (`scripts/statkit/`) before adding anything.
- Since Part 5 found the edge mostly lives pre-2022 and is erased by costs after, the diagnostic section should be used to state clearly WHETHER the spread is genuinely stationary/mean-reverting and how big the effect is — separating "the model finds real structure" from "the structure is not large enough to trade net of costs".
- Keep results/ git-ignored; commit only sources + tracked report artifacts meant to be tracked.
