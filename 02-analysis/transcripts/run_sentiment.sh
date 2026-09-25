#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript Sentiment [$CRISIS]: $name ==="
  uv run papermill "$NOTEBOOK_DIR/video-transcripts-sentiment-analysis.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p transcript_name "$name" --cwd "$NOTEBOOK_DIR" --log-output
done

echo "All transcript sentiment complete."
