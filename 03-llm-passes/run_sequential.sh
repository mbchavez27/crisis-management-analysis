#!/usr/bin/env bash
set -uo pipefail

CRISIS="ateneo-drowning"
INPUT_DIR="../01-preprocessing/output/$CRISIS/youtube"

# Find all phase CSVs, sort them numerically/alphabetically (01-onset -> 02-peak -> etc.)
phases=(\((ls "\)INPUT_DIR"/*.csv 2>/dev/null | sort))

if [ ${#phases[@]} -eq 0 ]; then
    echo "No phase CSVs found for $CRISIS!"
    exit 1
fi

for csv in "${phases[@]"; do
    phase="\((basename "\)csv" .csv)"
    echo "=================================================="
    echo " Running Phase: \(phase for crisis:\)CRISIS"
    echo "=================================================="
    
    uv run --no-sync python run_local.py \
        --crisis "$CRISIS" \
        --phases "$phase" \
        --passes 1 \
        --chunk-size 1 \
        --max-new-tokens 8192 \
        --load-in-4bit \
        --closing || {
            echo "Error encountered in phase $phase. Stopping."
            exit 1
        }
done

echo "All phases completed successfully!"
