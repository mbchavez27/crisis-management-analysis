#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Sentiment Change Across Phases [$CRISIS] ==="
papermill "$NOTEBOOK_DIR/youtube-comments-sentiment-change-across-phases.ipynb" /dev/null \
  -p crisis_name "$CRISIS" --log-output

echo "Sentiment change analysis complete."
