#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"
SRC="${SRC:-04-institutional_response-uaap_ban}"

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"
BACKUP_DIR="../../01-preprocessing/output/$CRISIS/youtube_backup"

for pass in grief-outrage basic-emotions action-demanded; do
  echo "=== Experiment: $pass [$SRC] ==="
  papermill "$NOTEBOOK_DIR/$pass-pass.ipynb" /dev/null \
    -p crisis_name "$CRISIS" \
    -p input_dir "$BACKUP_DIR" \
    -p sources "$SRC" \
    -p output_dir "../output/_experiments/$pass" \
    -p raw_dir "../output/_experiments/raw/$pass" \
    --cwd "$NOTEBOOK_DIR" --log-output
done

echo "Experiment complete."
