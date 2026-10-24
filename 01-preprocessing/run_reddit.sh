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

STAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK_DIR="$STAGE_DIR/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== Reddit [$CRISIS]: $phase ==="
  uv run --project "$STAGE_DIR" papermill "$NOTEBOOK_DIR/reddit-posts-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p phase_name "$phase" --cwd "$NOTEBOOK_DIR" --log-output
done

echo "All Reddit phases complete."
