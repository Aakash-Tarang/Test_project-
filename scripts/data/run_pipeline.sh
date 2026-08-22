#!/usr/bin/env bash
# Run the full data pipeline end-to-end: download -> clean -> QA -> summary.
# Usage: ./scripts/data/run_pipeline.sh [--source github|yahoo]
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="${1:-github}"
if [ "${1:-}" = "--source" ]; then SRC="${2:-github}"; fi

echo "==> 1/4 download ($SRC)"
python3 scripts/data/download.py --source "$SRC"

echo "==> 2/4 clean"
python3 scripts/data/clean.py

echo "==> 3/4 QA"
python3 scripts/data/qa.py

echo "==> 4/4 summary"
python3 scripts/analysis/make_data_summary.py

echo "==> pipeline complete"
