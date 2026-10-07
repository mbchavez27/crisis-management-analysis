#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$SCRIPT_DIR/notebooks"

echo "=== Statistical analysis [$CRISIS] ==="
echo "--- 01: merge and prep ---"
uv run papermill "$NOTEBOOK_DIR/01-merge-and-prep.ipynb" /dev/null \
  --cwd "$PROJECT_ROOT" --log-output

echo "--- 02: statistical analysis ---"
uv run papermill "$NOTEBOOK_DIR/02-statistical-analysis.ipynb" /dev/null \
  --cwd "$PROJECT_ROOT" --log-output

echo "Statistical analysis complete."
