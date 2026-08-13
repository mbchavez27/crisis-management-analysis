#!/usr/bin/env bash
set -euo pipefail

PHASES=(
  01-the_incident
  02-autopsy_request
  03.1-baldwin_apology
  03-public_outcry
  04-scholarships_withdrawal
  05-resignation_presscon
  06-townhall_podcast
  07-uaap_ban
  08-legal_probe
)

NOTEBOOK_DIR="$(cd "$(dirname "$0")" && pwd)/notebooks"

for phase in "${PHASES[@]}"; do
  echo "=== YouTube: $phase ==="
  papermill "$NOTEBOOK_DIR/youtube-comments-preprocessing.ipynb" /dev/null \
    -p batch_name "$phase" --log-output
done

echo "All YouTube phases complete."
