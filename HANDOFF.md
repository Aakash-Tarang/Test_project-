# HANDOFF — Next session: **Part 5 — Baseline Model Comparison (OLS/Ridge/Lasso/ElasticNet/PCA)**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency, **Part 4** correctness core (real-data spread backtest). Now: **Part 5**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

## 2. Environment (re-verify each session)
Run `export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 21/21 PASS.** If packages/tools are missing at session start (sandbox may reset installed packages), run `bash environment/setup.sh` first.
- pip PEP-668 → `pip install --user --break-system-packages`. No TeX engine (report = canonical LaTeX + HTML preview). Only pypi + GitHub reachable.
- Data persists in workspace snapshot (`data/processed/universe.csv`, 195 tickers, 764,896 rows, QA 10/10). If missing: `python3 scripts/data/clean.py --source yfinance && python3 scripts/data/qa.py`.
- Build: `cmake -S src -B build && cmake --build build -j2`. Binaries: `build/statarbsim` (--version/--test/--demo/--backtest), `build/microbench`.

## 3. Part 4 delivered (verified 21/21)
- **Causal, walk-forward spread backtest engine** `src/engine/spread_backtest.h` (clean, dependency-free, single-header): rolling window-W regression of target log-price on basket log-prices over window ending at `t-1` (hedge betas causal); standardized residual `z_t` triggers market-neutral entry (`|z|≥z_entry`) executed from the **next bar** (one-bar delay, no lookahead); exit when `|z|<z_exit`. Fixed hedge weights at entry, gross notional normalized to $1 via `g=1+Σ|β_i|`; dollar-neutral `ret = (pos/g)·(−lr_target + Σβ_i·lr_i)`. Cost charged `cost_one_way·g` on each open+close (both legs). Summary incl. annualized Sharpe (sqrt-252), trades, avg holding, cost(log), maxDD. Output per-bar CSV with `date,pos,t,z,lognav_gross,lognav_net`.
- `src/main.cpp` `--backtest <csv> <out> [window entry exit cost]` CLI; `src/data/csv_loader.h` minimal loader that skips the leading date column. Real basket via `scripts/data/export_engine_input.py` → `data/processed/engine/ADBE.csv` (target ADBE; CRM/ADSK/INTU), 4 assets × 4,190 bars (git-ignored).
- **16 tests / 2,802 checks, 0 failed** (`--test`), incl. new no-lookahead (`test_backtest_one_bar_delay_no_lookahead`), too-short (`test_backtest_no_trade_when_too_short`), cost monotonicity (`test_backtest_costs_never_improve_net`). All prior tests pass.
- Real-data run (window 120, entry 2.0, exit 0.5): **198 trades, avg hold 16.2 bars, cost(log)=0.4390; gross ann_ret 0.0302 / Sharpe 0.385; net (10 bps) ann_ret 0.0038 / Sharpe 0.050; maxDD_net 0.4188**. Cost sweep 0/5/10/20/50 bps shows the daily-frequency edge is nearly erased by 10 bps costs and negative beyond — an honest, report-worthy finding motivating Part 5.
- Pipeline: `scripts/analysis/capture_backtest.py` (runs cost sweep → `results/tables/backtest_summary.csv`, copies 10 bps series → `results/backtests/ADBE_main.csv`) + `scripts/plotting/render_backtest.py` (→ `results/figures/{equity_curve,cost_sensitivity}.png`, copied to `report/figures/`; writes `report/tables/backtest_base.tex`). Both wired into `report/build_report.py`; check.sh gains 3 Part-4 checks. Report §7 (Backtest Results) fully written (methodology/no-lookahead, square-root cost model \cite{almgren2005}, base-basket results, tables/figures). `results/*` outputs are git-ignored (reproducible).

## 4. YOUR TASK — Part 5: Baseline Model Comparison (OLS/Ridge/Lasso/ElasticNet/PCA)
Implement and honestly compare a set of baseline predictive models on the spread/basket data, evaluating no-lookahead signal quality and (where sensible) backtested net performance. Goal: establish which hedger/predictor variant earns the best *net-of-cost* signal for the spread stat-arb, and fill the Part 3 stubs in `src/model/basket_selector.h` (OLS/Ridge already present; Lasso/ElasticNet/PCA stubbed).

### 4.1 Deliverables (suggested)
1. **`src/model/basket_selector.h`** — complete the stubbed Lasso, ElasticNet and PCA methods; keep Ridge & OLS. Decide (and document) the primary use: these predict/hedge basket weights; keep them dependency-free and deterministic.
2. **Baseline evaluation harness** — for each model, on a fixed panel (reuse the `data/processed/engine/*` wide exports or a new one): rolling/expanding walk-forward fit, produce a comparable signal (standardized residual z or predicted return), and score by predictive metrics (e.g. IC / rank-IC vs realized next-period target move, or the realized hedged-spread Sharpe net of the Part-4 cost model). Keep the no-lookahead one-bar-delay discipline and reuse `LatencyHist` if you time per-bar fits.
3. **PCA variant** — principal-components hedge/rotation on the basket covariance (eig already in `model/dense_la.h`) vs pure regression basket; report what changes.
4. **Tests** — extend `test/engine_tests.cpp` for the new model paths (fit determinism, Ridge≈OLS at λ→0, Lasso sparsifies, PCA variance ordering). Keep `--test` growing; `environment/check.sh` total must not regress below 21 and should grow.
5. **Real comparison run + figures/table** — one or more comparison figures/tables into `results/figures` + `report/tables`, driven by new scripts and wired into `report/build_report.py`. Honest results (some models may not beat OLS / may not be net profitable on this data).
6. **Report** — write the model-comparison section prose with methodology (cross-val scheme, metric choice, hyperparameter selection for the penalties, and how regularization helps/hurts given the daily-frequency cost drag found in Part 4).

### 4.2 Exit gate (must pass before done)
- [ ] OLS/Ridge/Lasso/ElasticNet/PCA all implemented and compared in a no-lookahead walk-forward setting; differences reported honestly (table/figure from the live run).
- [ ] Hyperparameter selection for penalties documented (what λ/α grids were used and how chosen).
- [ ] New unit tests for model paths pass; all prior tests still pass; `environment/check.sh` ≥21 and grows if practical.
- [ ] Report section written and regenerated (HTML preview; LaTeX canonical).
- [ ] `git status` clean after commit; `HANDOFF.md` written for **Part 6** (per PLAN.md).

### 4.3 Hints
- Prefer Python (`scripts/analysis`, `scripts/plotting`) + numpy/sklearn for the broad model sweep and cross-val; the C++ `basket_selector.h` is the in-engine implementation to complete for consistency/tests. Cite in report what each does.
- The Part-4 finding (edge ~erased at 10 bps) means Part 5 should evaluate signals **net of the same cost model**; if PCA/regularization improves persistence/holding it can lift net Sharpe even if gross looks similar. Frame comparisons on net metrics.
- Reuse existing report/plot/table/capture conventions. Keep `results/*` git-ignored; commit only sources + the regenerated tracked report assets that are meant to be tracked.
