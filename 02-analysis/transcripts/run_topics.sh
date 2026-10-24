#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_DIR="$(cd "$NOTEBOOK_DIR/.." && pwd)"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript Topics [$CRISIS]: $name ==="
  uv run --project "$STAGE_DIR" papermill "$NOTEBOOK_DIR/video-transcripts-topic-modeling.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p transcript_name "$name" --cwd "$NOTEBOOK_DIR" --log-output
done

echo "All transcript topic phases complete."
