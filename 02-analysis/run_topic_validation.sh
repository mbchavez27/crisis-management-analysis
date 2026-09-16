#!/usr/bin/env bash
set -euo pipefail

echo "=== Generating cross-phase review sheet ==="
papermill notebook/youtube-comments-topic-validation.ipynb /dev/null \
  -p mode generate \
  -p review_csv_path "../../02-analysis/output/topic-review-sheet.csv" \
  --log-output

echo ""
echo "Review sheet: 02-analysis/output/topic-review-sheet.csv"
echo ""
echo ">>> OPEN THE CSV AND FILL: human_label, action, merge_to, notes <<<"
echo ">>> Actions: confirm / rename / merge / reject <<<"
echo ""
read -p "Press Enter after validating the CSV (or Ctrl+C to cancel)... "

echo ""
echo "=== Applying validated decisions ==="
papermill notebook/youtube-comments-topic-validation.ipynb /dev/null \
  -p mode apply \
  -p review_csv_path "../../02-analysis/output/topic-review-sheet.csv" \
  --log-output

echo "All phase CSVs updated."
