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
  rm -rf build && cmake -S src -B build >/dev/null 2>&1 && cmake --build build -j2 >/dev/null 2>&1
  chk "CMake configure+build" $?
  if [ -x build/statarbsim ]; then ./build/statarbsim --version >/dev/null 2>&1; chk "engine smoke run" $?; fi
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
