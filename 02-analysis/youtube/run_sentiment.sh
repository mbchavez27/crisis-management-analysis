#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

PHASES=(
  01-onset
  02-confirmation
  03-spread
  04-reckoning
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube Sentiment [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/youtube-comments-sentiment-analysis.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --cwd "$NOTEBOOK_DIR" --log-output
done

echo "All YouTube sentiment phases complete."
