#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK_DIR="$STAGE_DIR/notebooks"

echo "=== Summary / Insights [$CRISIS] ==="
uv run --project "$STAGE_DIR" papermill "$NOTEBOOK_DIR/summary.ipynb" /dev/null \
  -p crisis_name "$CRISIS" --cwd "$NOTEBOOK_DIR" --log-output

echo "Summary complete."
