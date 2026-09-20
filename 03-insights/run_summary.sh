#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

echo "=== Summary / Insights [$CRISIS] ==="
papermill "$NOTEBOOK_DIR/summary.ipynb" /dev/null \
  -p crisis_name "$CRISIS" --log-output

echo "Summary complete."
