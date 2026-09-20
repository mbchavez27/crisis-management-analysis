#!/usr/bin/env bash
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
REVIEW_CSV="$(cd "$(dirname "$0")/../../02-analysis/output/$CRISIS" && pwd)/topic-review-sheet.csv"

echo "=== Generating cross-phase review sheet ==="
papermill "$NOTEBOOK_DIR/youtube-comments-topic-validation.ipynb" /dev/null \
  -p mode generate \
  -p review_csv_path "$REVIEW_CSV" \
  --cwd "$NOTEBOOK_DIR" --log-output

echo ""
echo "Review sheet: $REVIEW_CSV"
echo ""
echo ">>> OPEN THE CSV AND FILL: human_label, action, merge_to, notes <<<"
echo ">>> Actions: confirm / rename / merge / reject <<<"
echo ""
read -p "Press Enter after validating the CSV (or Ctrl+C to cancel)... "

echo ""
echo "=== Applying validated decisions ==="
papermill "$NOTEBOOK_DIR/youtube-comments-topic-validation.ipynb" /dev/null \
  -p mode apply \
  -p review_csv_path "$REVIEW_CSV" \
  --cwd "$NOTEBOOK_DIR" --log-output

echo "All phase CSVs updated."
