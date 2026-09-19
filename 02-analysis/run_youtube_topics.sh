#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

PHASES=(
  01-onset
  02-confirmation
  03-spread
  04-reckoning
  05-lingering
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebook"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube Topics [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/youtube-comments-topic-modeling.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --log-output
done

echo "All YouTube topic phases complete."
