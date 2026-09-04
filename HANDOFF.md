# HANDOFF — Next session: **Part 3 — C++ Engine Skeleton + Latency Instrumentation**

> Read this entire file before doing anything.

---

## 1. Context
Incremental **Multi-Asset Statistical Arbitrage** project. Done: **Part 0** (plan/environment), **Part 1** (data + universe), **Part 2** (statistical toolkit). Your task now: **Part 3**. Work **only** on `arena/01a029cc-test-project`; commit/push only there. Master plan: `PLAN.md`.

---

## 2. Environment (re-verify with `environment/check.sh`)
**IMPORTANT — this sandbox does NOT persist installed packages between sessions.** At the start of a session, if `check.sh` shows failures, re-run:
```bash
export PATH="$HOME/.local/bin:$PATH"
bash environment/setup.sh          # reinstalls python stack + cmake (user-level)
bash environment/check.sh          # currently 15/15 checks, must pass
```
- `g++`/`make` are system-provided. `cmake` is pip-installed into `~/.local/bin` (add to PATH).
- pip is PEP-668 → always `python3 -m pip install --user --break-system-packages ...`.
- **Reachable:** pypi, github.com, api.github.com, codeload. **Blocked:** Yahoo/stooq/Nasdaq, apt, conda, rust, TeX engines.
- **No TeX engine** → report is canonical LaTeX (`report/report.tex`), built via `report/build_report.py` which regenerates tables/figures and always writes an HTML preview. Compile the PDF wherever a TeX engine exists.
- **Data persistence:** `data/processed/`, `results/` are git-ignored but usually persist via the workspace snapshot. If `data/processed/universe.csv` is missing, the user's committed raw Yahoo data lives at `data/raw/yfinance/*.csv` (195 files, TRACKED in git); run `python3 scripts/data/clean.py --source yfinance && python3 scripts/data/qa.py`.

## 3. Primary dataset now (from Part 1 refresh this session)
- **Yahoo Finance** daily OHLCV + fully-adjusted Adj Close, user-downloaded 2026-09-04, committed under `data/raw/yfinance/`.
- **195 tickers**, 2010-01-04 .. 2026-09-01 (~16.7 yr), 764,896 cleaned rows in `data/processed/universe.csv`, QA **10/10**.
- 195 < 200 spec target → disclosed in report §Data. NYSE (501-name, 2010-2016) obtainable in-sandbox via `--source github` as an auxiliary cross-section.
- `sp500_symbols.csv`/`sector_map.json` (505 names with GICS sectors) are committed under `scripts/data/`.

---

## 4. YOUR TASK — Part 3: C++ Engine Skeleton + Latency Instrumentation

### 4.1 Goal
Stand up the **low-latency, dependency-free, latency-instrumented C++ core** per spec §5. Build the required class architecture with correctness tests from day one. C++ is the simulation core — Python stays research/plotting only.

### 4.2 Environment decision (document in code + report)
Spec §5.1 wants Eigen for SIMD-friendly aligned contiguous matrices, BUT **this sandbox cannot fetch Eigen** (apt/conda blocked) and cannot run external downloads except GitHub/pypi. Eigen is header-only and available on GitHub — **you MAY vendor a minimal Eigen into `third_party/eigen/`** IF reachable, otherwise implement hand-rolled dense linear algebra (a small matrix class + LDLT/LU solve + eigendecomposition via symmetric QR/Jacobi) that is contiguous and cache-friendly. The project rule is **dependency-free at build time**: no `#include` that isn't in the repo or the system toolchain. Justify whichever route in the code comments and the report.

### 4.3 Required classes (mirror spec §5.2; keep the repo structure under `src/`)
Implement in `src/data, src/model, src/signal, src/portfolio, src/engine`:
- `src/data/market_data_buffer.{h,cpp}` — **SoA ring buffer** over `std::vector` (separate contiguous `timestamps_`, `prices_`, `volumes_`), fixed capacity, `push()` O(1) no-alloc, contiguous `std::span` views, no dynamic allocation after construction. Unit test eviction/ring-wrap.
- `src/model/rolling_regression.{h,cpp}` — rolling linear regression with **O(n_features²) incremental rank-1 update** of `XᵀX`/`XᵀY` (Welford/online normal equations), window eviction via a flat row-major `feature_history_` ring buffer; expose `beta()`, `r_squared()`, `residual()`, `residual_zscore()`. `ridge_lambda` support. Test against brute-force recomputation to tolerance.
- `src/model/basket_selector.{h,cpp}` — interface + OLS/Ridge at minimum this Part (Lasso/ElasticNet/PCA paths come in Part 5 — stub them cleanly now with an enum so the API is stable).
- `src/signal/signal_generator.{h,cpp}` — entry/exit z bands + half-life-aware gate; emits a `TradeSignal`.
- `src/portfolio/portfolio_book.{h,cpp}` — flat `std::vector<double>` positions indexed by asset_id, gross/net exposure. (Full risk controls/VaR in Part 7; keep the flat-table design.)
- `src/engine/backtest_engine.{h,cpp}` — orchestrates the pipeline with **per-stage latency histograms** (p50/p95/p99) via `std::chrono::high_resolution_clock` or `rdtsc`; `LatencyReport latencyStats()`.
- `src/latency_hist.{h,cpp}` (or in engine) — a fixed-capacity histogram over a flat array, no allocation, computing p50/p95/p99.

### 4.4 Build & test
- `src/CMakeLists.txt` currently builds a single `statarbsim` executable from `src/main.cpp`. Extend it: C++17, `-O2 -Wall -Wextra`, and a `--test` mode that runs the hand-rolled test harness.
- Add `test/minimal_test.hpp` — a tiny `RUN_TEST(name){...}` macro + assertion helpers, and `src/main.cpp` `--test` flag runs all tests. (No external framework needed; spec accepts Catch2/GoogleTest but we are dependency-light.)
- Unit tests required now: rolling-regression incremental == brute force (tolerance ~1e-8), ring-buffer eviction correctness, SoA push/span correctness, window-not-yet-full, no-allocation in hot path (override `operator new` / count allocations in a test), beta under `ridge_lambda`, residual/zscore math.
- `bench/` microbenchmarks (compile + runnable, real numbers): `vector` vs `deque` rolling window, SoA vs AoS. Report measured throughput/cache-miss where possible (`perf stat` if available; otherwise cycle-count with `rdtsc` or `std::chrono`).

### 4.5 Report + figures
- Write report §9 text headers and, from the **real** benchmark run, generate latency histogram + cache/SoA figures into `results/figures/` (a `scripts/plotting/render_engine_*.py` or C++-emitted CSV that the report builder plots). Follow the convention: numbers must come from an actual run, not hand-typed.
- Report the environment decision (§Eigen vs hand-rolled), the SoA/cache rationale, and the no-alloc verification in §9 prose.

### 4.6 Exit gate (must pass before done)
- [ ] C++ engine compiles (CMake) and `--test` runs **all** unit tests passing.
- [ ] Incremental rolling regression matches brute-force recomputation to stated tolerance.
- [ ] No dynamic allocation verified in the hot path (allocation-count test passes).
- [ ] Latency histograms report real p50/p95/p99 numbers per stage (no placeholders).
- [ ] `bench/` runs and records real vector-vs-deque and SoA-vs-AoS numbers.
- [ ] Report §9 section drafted with the architecture rationale + environment decision; latency & bench figures generated from real runs and referenced.
- [ ] `environment/check.sh` passes (15 existing + new C++ `--test` check → ≥16).
- [ ] `git status` clean after commit; new `HANDOFF.md` written for **Part 4 (Correctness Core of the Engine: walk-forward loop, costs, no-lookahead, edge cases)**.

### 4.7 Hints / pitfalls
- **Ring buffers over flat arrays, not `std::deque`/`std::list`** — spec §5.1 is explicit (cache lines, spatial locality, prefetch). Justify in comments.
- **SoA over AoS** for the hot per-bar loop. Justify + benchmark.
- **Avoid vtables/heap alloc in the per-tick path** — templates/CRTP if polymorphism is needed; pre-allocate everything.
- Rolling regression: maintain running `XᵀX`, `XᵀY`, and sums; on window eviction subtract the leaving row's outer product (rank-1), on push add the new row's outer product — never refit from scratch in the hot path. Because a plain online inverse can drift, you may refit `(XᵀX)`'s inverse via Cholesky at each step only if O(n³) is acceptable for small n, or maintain the inverse via Sherman–Morrison; document the numerical approach and its accuracy test vs brute force.
- Keep the C++ self-contained and fast to compile (single translation unit where practical) to keep iteration fast on 2 cores.
- Do NOT implement the full multi-basket portfolio/VaR yet (Part 7) — but design `PortfolioBook` so it can be extended.

---

## 5. Ritual (every Part)
1. Implement → run tests + `environment/check.sh` → update report → commit → push to `arena/01a029cc-test-project`.
2. Overwrite `HANDOFF.md` for the next Part.
3. Keep conventions (paths, report builder, regeneration). Don't restructure existing files.
