#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/../03-insights/notebooks"

echo "=== Summary / Insights ==="
papermill "$NOTEBOOK_DIR/summary.ipynb" /dev/null \
  --log-output

echo "Summary complete."
