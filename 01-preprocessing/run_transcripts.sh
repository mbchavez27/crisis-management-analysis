#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript [$CRISIS]: $name ==="
  papermill "$NOTEBOOK_DIR/video-transcripts-preprocessing.ipynb" /dev/null \
    -p crisis_name "$CRISIS" -p transcript_name "$name" --log-output
done

echo "All transcripts complete."
