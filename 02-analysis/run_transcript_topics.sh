#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPTS=(
  ateneo_presscon
  coach_tab
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebook"

for name in "${TRANSCRIPTS[@]}"; do
  echo "=== Transcript Topics: $name ==="
  papermill "$NOTEBOOK_DIR/video-transcripts-topic-modeling.ipynb" /dev/null \
    -p transcript_name "$name" --log-output
done

echo "All transcript topic phases complete."
