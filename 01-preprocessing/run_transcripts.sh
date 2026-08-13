#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript: $name ==="
  papermill "$NOTEBOOK_DIR/video-transcripts-preprocessing.ipynb" /dev/null \
    -p transcript_name "$name" --log-output
done

echo "All transcripts complete."
