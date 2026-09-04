#!/usr/bin/env bash
# Exit-gate sanity check. Run before declaring any Part complete.
# Fails (non-zero) if a core requirement is missing.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="python3"
export PATH="$HOME/.local/bin:$PATH"
pass=0; fail=0
chk() { # chk <description> <result(0=ok,else=fail)>
  if [ "$2" -eq 0 ]; then echo "  [PASS] $1"; pass=$((pass+1)); else echo "  [FAIL] $1"; fail=$((fail+1)); fi
}

echo "== Toolchain =="
command -v g++      >/dev/null 2>&1; chk "g++ present" $?
command -v make     >/dev/null 2>&1; chk "make present" $?
command -v cmake    >/dev/null 2>&1; chk "cmake on PATH" $?
command -v "$PY"    >/dev/null 2>&1; chk "python3 present" $?

echo "== Python packages =="
"$PY" -c "import numpy, pandas, scipy, statsmodels, matplotlib, seaborn, sklearn, joblib" >/dev/null 2>&1; chk "scientific stack imports" $?
"$PY" -c "import reportlab" >/dev/null 2>&1; chk "reportlab present" $?

echo "== C++ build smoke (if src exists) =="
if [ -f src/CMakeLists.txt ]; then
  cmake -S src -B build >/dev/null 2>&1 && cmake --build build -j2 >/dev/null 2>&1
  chk "CMake configure+build" $?
  if [ -x build/statarbsim ]; then
    ./build/statarbsim --version >/dev/null 2>&1; chk "engine smoke run" $?
    ./build/statarbsim --test >/dev/null 2>&1; chk "C++ unit tests pass (--test)" $?
    ./build/statarbsim --demo >/dev/null 2>&1; chk "engine latency demo runs" $?
  fi
  "$PY" scripts/analysis/capture_engine_bench.py >/dev/null 2>&1
  [ -f results/tables/engine_bench.csv ]; chk "engine bench CSV present" $?
else
  echo "  [SKIP] no src/CMakeLists.txt yet"
fi

echo "== Report build driver =="
if [ -f report/build_report.py ]; then
  "$PY" report/build_report.py >/dev/null 2>&1; chk "report/build_report.py runs" $?
  [ -f report/report.html ]; chk "report HTML preview produced" $?
else
  echo "  [SKIP] no report/build_report.py yet"
fi

echo "== Spread backtest (Part 4) =="
"$PY" scripts/analysis/capture_backtest.py >/dev/null 2>&1; chk "backtest capture runs" $?
[ -f results/tables/backtest_summary.csv ]; chk "backtest summary table present" $?
[ -f results/figures/equity_curve.png ]; chk "equity curve figure present" $?

echo "== Model comparison (Part 5) =="
[ -f results/tables/model_compare_summary.csv ]; chk "model comparison summary present" $?
[ -f results/figures/model_compare_sharpe.png ]; chk "model comparison Sharpe figure present" $?
[ -f report/tables/model_compare.tex ]; chk "model comparison table present" $?

echo "== Portfolio book + risk controls (Part 7) =="
"$PY" scripts/analysis/capture_portfolio.py >/dev/null 2>&1; chk "portfolio driver runs" $?
[ -f results/tables/portfolio_summary.csv ]; chk "portfolio summary present" $?
[ -f results/figures/portfolio_equity.png ]; chk "portfolio equity figure present" $?
[ -f report/tables/portfolio.tex ]; chk "portfolio table present" $?

echo "== Kalman time-varying hedge (Part 8) =="
"$PY" scripts/test_kalman.py >/dev/null 2>&1; chk "kalman unit tests pass" $?
[ -f results/tables/kalman_summary.csv ]; chk "kalman comparison summary present" $?
[ -f results/figures/kalman_sharpe.png ]; chk "kalman Sharpe figure present" $?
[ -f report/tables/kalman.tex ]; chk "kalman table present" $?

echo "== Diagnostics (Part 6) =="
"$PY" scripts/analysis/run_diagnostics.py >/dev/null 2>&1; chk "diagnostics driver runs" $?
[ -f results/tables/diagnostics_summary.csv ]; chk "diagnostics summary present" $?
[ -f results/figures/diagnostics_halflife.png ]; chk "diagnostics half-life figure present" $?
[ -f report/tables/diagnostics.tex ]; chk "diagnostics table present" $?

echo "== Nonlinear signal extraction + RESET (Part 9) =="
"$PY" scripts/test_nonlinear.py >/dev/null 2>&1; chk "nonlinear pipeline unit tests pass" $?
[ -f results/tables/nonlinear_summary.csv ]; chk "nonlinear forecast summary present" $?
[ -f results/tables/nonlinear_reset.csv ]; chk "nonlinear RESET table present" $?
[ -f results/tables/nonlinear_trading.csv ]; chk "nonlinear gated-trading table present" $?
[ -f results/figures/nonlinear_ic.png ]; chk "nonlinear IC figure present" $?
[ -f results/figures/nonlinear_learning.png ]; chk "nonlinear learning-curve figure present" $?
[ -f report/tables/nonlinear.tex ]; chk "nonlinear table present" $?
"$PY" -c "
import sys,os,csv
p=os.path.join('results','tables','nonlinear_trading.csv')
rows=list(csv.reader(open(p)))[1:]
linear_full=[r for r in rows if 'linear-gate' in r[0] and 'full' in r[1]]
base_full=[r for r in rows if 'z-threshold' in r[0] and 'full' in r[1]]
if linear_full and base_full and float(linear_full[0][2])>float(base_full[0][2])+1e-9:
    sys.exit(1)  # nonlinear gate must NOT beat baseline net of costs (honest gate)
" >/dev/null 2>&1; chk "nonlinear-gate does not outclaim baseline net Sharpe" $?

echo "== statkit (Part 2) =="
"$PY" scripts/run_statkit_tests.py >/dev/null 2>&1; chk "statkit unit tests pass" $?

echo "== Data pipeline (Part 1) =="
if [ -f data/processed/universe.csv ]; then
  chk "cleaned universe present" 0
  "$PY" scripts/data/qa.py >/dev/null 2>&1; chk "data QA passes (qa.py)" $?
  [ -f results/tables/data_summary.csv ]; chk "data summary table present" $?
  [ -f results/figures/universe_by_year.png ]; chk "data figures present" $?
else
  echo "  [SKIP] no data/processed/universe.csv yet (run scripts/data/download.py + clean.py)"
fi

echo ""
echo "== RESULT: $pass passed, $fail failed =="
[ "$fail" -eq 0 ]
