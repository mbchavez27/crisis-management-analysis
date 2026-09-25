#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Sentiment Change Across Phases [$CRISIS] ==="
uv run papermill "$NOTEBOOK_DIR/youtube-comments-sentiment-change-across-phases.ipynb" /dev/null \
  -p crisis_name "$CRISIS" --cwd "$NOTEBOOK_DIR" --log-output

echo "Sentiment change analysis complete."
