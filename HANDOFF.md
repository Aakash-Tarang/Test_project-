# HANDOFF — Next session: **Part 4 — Correctness Core of the Engine**

> Read this entire file first.

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** plan/env, **Part 1** data (Yahoo 195-name, 2010-2026), **Part 2** statkit, **Part 3** C++ engine skeleton + latency. Now: **Part 4**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Plan: `PLAN.md`.

## 2. Environment (re-verify each session)
Run `export PATH="$HOME/.local/bin:$PATH"` then `bash environment/check.sh`. **Now 18/18 PASS.** If packages/tools are missing at session start (sandbox may reset installed packages), run `bash environment/setup.sh` first.
- pip PEP-668 → `pip install --user --break-system-packages`. No TeX engine (report = canonical LaTeX + HTML preview). Only pypi + GitHub reachable.
- Data persists in workspace snapshot (`data/processed/universe.csv`, 195 tickers, 764,896 rows, QA 10/10). If missing: `python3 scripts/data/clean.py --source yfinance && python3 scripts/data/qa.py` (raw Yahoo CSVs are tracked under `data/raw/yfinance/`).
- Build: `cmake -S src -B build && cmake --build build -j2`. Binaries: `build/statarbsim` (--version/--test/--demo), `build/microbench`.

## 3. Part 3 delivered (verified 18/18)
- C++ engine (dependency-free) under `src/{data,model,signal,portfolio,engine}`:
  - `data/market_data_buffer.h` — SoA, fixed-capacity, contiguous, no-alloc push.
  - `model/dense_la.h` — row-major contiguous dense layer: SPD/ridge Cholesky solve + cyclic-Jacobi symmetric eig (for PCA later).
  - `model/rolling_regression.h` — incremental rank-1 windowed regression, flat row-major ring buffer, pre-allocated Cholesky scratch, **no allocation in update()/compute()** (verified).
  - `model/basket_selector.h` — OLS/Ridge implemented; Lasso/ElasticNet/PCA stubbed (Part 5).
  - `signal/signal_generator.h`, `portfolio/portfolio_book.h` (flat table; full risk in Part 7), `engine/latency_hist.h`, `engine/backtest_engine.h`.
- `test/minimal_test.hpp` + `test/engine_tests.cpp`: **13 tests / 1006 checks, 0 failed** (`build/statarbsim --test`).
- `bench/microbench.cpp`: vector-vs-deque (~1.15x) and SoA-vs-AoS (~1.03x) measured honestly.
- Report §9 written; figures `results/figures/{latency_hist,cache_bench}.png` + `report/tables/engine_bench.tex` from live runs via `scripts/analysis/capture_engine_bench.py` + `scripts/plotting/render_engine_results.py`, wired into `report/build_report.py`.

## 4. YOUR TASK — Part 4: Correctness Core of the Engine
Make the simulation loop **correct, walk-forward, cost-aware, and lookahead-free** (spec §4.2), and harden the engine with edge-case tests.

### 4.1 Deliverables
1. **Walk-forward backtest loop** in `BacktestEngine` (real, streaming the `MarketDataBuffer` history, not the synthetic demo): for each bar at time t, generate a signal from data through t but **execute at t+1** (no lookahead). Document train/validation/test date split in the report §7 (e.g. train 2010-2018, val 2019-2021, test 2022+ — adapt to the 2010-2026 Yahoo data).
2. **Cost model** (spec §4.2): empirically estimated per-name bid-ask spread (use open-high-low proxy or a documented flat spread per liquidity tier), commission, and market-impact via the square-root model `impact ∝ σ√(Q/V)`. Report gross AND net of costs.
3. **Portfolio/equity accounting**: PortfolioBook already stores flat positions + equity history; implement per-bar P&L mark-to-market, position sizing from the residual z-score / inverse-vol, turnover tracking.
4. **Edge cases & no-lookahead tests**: window-not-yet-full already tested; add (a) delisting / missing bar mid-backtest, (b) signal at last available bar can't be filled (no future), (c) cost applied on both entry and exit, (d) a brute "delayed-by-one" cross-check that signals built on time-t info are only tradeable at t+1.
5. **Unit tests** for all of the above (extend `test/engine_tests.cpp`, still via `--test`).
6. **Real run + figures**: run the engine over the actual `universe.csv` for a representative basket to produce an equity curve gross vs net (a real result, even if only for one target/basket — full multi-basket is Part 7). Save to `results/backtests/` and plot via a new script into `results/figures/` + report §7. Report numbers must come from the run.
7. **Report**: write §7 (Backtest Results) prose for the walk-forward methodology + the cost model derivation (square-root impact), and include the gross/net equity figure and a small table.

### 4.2 Exit gate (must pass before done)
- [ ] Walk-forward loop is lookahead-free by construction; add a specific no-lookahead unit test that fails if a bar t signal is filled at t.
- [ ] Cost model implemented (spread + commission + square-root market impact); gross and net P&L both computed.
- [ ] Edge-case tests (delisting/missing bar, cost both sides, stale last signal) pass.
- [ ] All prior tests still pass; `environment/check.sh` passes (18 + any new → ≥18); new checks added for Part 4 tests if practical.
- [ ] Real-data engine run produces a gross-vs-net equity figure and backtest metrics saved to `results/`; report §7 updated and regenerated.
- [ ] `git status` clean after commit; `HANDOFF.md` written for **Part 5 (Baseline Model Comparison: OLS/Ridge/Lasso/ElasticNet/PCA)**.

### 4.3 Hints
- Keep the C++ dependency-free and fast. A per-bar solve is O(n_feat³) with tiny n_feat — fine.
- The engine should consume the committed data cleanly. You may add a small C++ loader for the `data/processed/universe.csv` long panel, or have the engine read a wide matrix you export from Python. Prefer: a `scripts/data/export_engine_input.py` that writes a compact wide binary/CSV (`target, features...`) for a chosen basket to `data/processed/engine/` (git-ignored), and a C++ loader. Document the interface in §7/§9.
- Timing: latency instrumentation already exists; reuse `LatencyHist` for the per-bar stages in the real loop.
- Do NOT build the multi-basket/risk-parity portfolio or VaR yet (Part 7) — but keep PortfolioBook/engine structured so it extends.
