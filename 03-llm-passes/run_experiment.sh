#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"
SRC="${SRC:-02-immediate_reaction_extended}"

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
    -p sample_n 2000 \
    -p sample_seed 42 \
    --cwd "$NOTEBOOK_DIR" --log-output
done

echo "Experiment complete."
