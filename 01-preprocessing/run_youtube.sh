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

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/youtube-comments-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p batch_name "$phase" --log-output
done

echo "All YouTube phases complete."
