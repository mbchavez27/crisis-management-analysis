#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

PHASES=(
  00-unrelated
  01-onset
  02-confirmation
  03-spread
  04-reckoning
  05-lingering
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== Reddit [$CRISIS]: $phase ==="
  papermill "$NOTEBOOK_DIR/reddit-posts-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --log-output
done

echo "All Reddit phases complete."
