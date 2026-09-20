#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript Topics [$CRISIS]: $name ==="
  papermill "$NOTEBOOK_DIR/video-transcripts-topic-modeling.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p transcript_name "$name" --log-output
done

echo "All transcript topic phases complete."
