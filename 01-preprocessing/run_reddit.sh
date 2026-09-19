#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

PHASES=(
  00-unrelated
  01-immediate_reaction_24h
  02-immediate_reaction_extended
  03-escalation
  04-institutional_response
  05-aftermath
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== Reddit [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/reddit-posts-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --log-output
done

echo "All Reddit phases complete."
