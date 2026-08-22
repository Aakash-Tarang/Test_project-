#!/usr/bin/env bash
# Bootstrap the full toolchain for this project.
# Idempotent. Safe to re-run every session.
# NOTE: pip is PEP-668 externally-managed here, so we always use --user --break-system-packages.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="python3"

echo "==> [1/3] Installing Python scientific stack (user-level)..."
$PY -m pip install --user --break-system-packages --quiet \
    numpy pandas scipy statsmodels matplotlib seaborn scikit-learn joblib \
    yfinance reportlab pyyaml 2>&1 | tail -2 || true

echo "==> [2/3] Installing CMake (user-level)..."
$PY -m pip install --user --break-system-packages --quiet cmake 2>&1 | tail -2 || true

echo "==> [3/3] Verifying C++ toolchain..."
command -v g++ make || { echo "FATAL: need g++ and make"; exit 1; }
g++ --version | head -1
make --version | head -1

export PATH="$HOME/.local/bin:$PATH"
cmake --version | head -1 || echo "WARNING: cmake not on PATH"

echo "==> Bootstrap complete. Run ./environment/check.sh to verify."
