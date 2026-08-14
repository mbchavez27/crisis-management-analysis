#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebook"

echo "=== Sentiment Change Across Phases ==="
papermill "$NOTEBOOK_DIR/youtube-comments-sentiment-change-across-phases.ipynb" /dev/null \
  --log-output

echo "Sentiment change analysis complete."
