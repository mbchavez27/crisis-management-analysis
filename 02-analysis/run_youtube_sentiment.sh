#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

PHASES=(
  01-immediate_reaction_24h
  02-immediate_reaction_extended
  03-escalation
  04-institutional_response
  05-aftermath
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebook"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube Sentiment [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/youtube-comments-sentiment-analysis.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --log-output
done

echo "All YouTube sentiment phases complete."
