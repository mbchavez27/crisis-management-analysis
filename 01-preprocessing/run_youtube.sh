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

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube [$CRISIS]: $phase ==="
  uv run papermill "$NOTEBOOK_DIR/youtube-comments-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p batch_name "$phase" --cwd "$NOTEBOOK_DIR" --log-output
done

echo "All YouTube phases complete."
