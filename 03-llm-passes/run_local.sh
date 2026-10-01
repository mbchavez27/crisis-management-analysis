#!/usr/bin/env bash
# Local-LLM coding pass over EVERY crisis and EVERY phase CSV, all rows (no sampling).
# Crises/phases are discovered from 01-preprocessing/output/<crisis>/youtube/<phase>.csv
#
# Usage:
#   bash 03-llm-passes/run_local.sh
#   OVERWRITE=1 bash 03-llm-passes/run_local.sh        # discard checkpoints, start fresh
#   CRISES=ateneo-drowning bash 03-llm-passes/run_local.sh   # restrict to some crises (comma-separated)
# Re-running the same command resumes from checkpoints.
set -uo pipefail
export UV_NO_SYNC="${UV_NO_SYNC:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_ROOT="$SCRIPT_DIR/../01-preprocessing/output"

PASSES="${PASSES:-1}"
MODEL="${MODEL:-aisingapore/Gemma-SEA-LION-v3-9B}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
OVERWRITE="${OVERWRITE:-}"
CRISES="${CRISES:-}"

ARGS=(--passes "$PASSES" --model "$MODEL" --chunk-size "$CHUNK_SIZE" --max-new-tokens "$MAX_NEW_TOKENS")
[[ "$LOAD_IN_4BIT" == "1" ]] && ARGS+=(--load-in-4bit)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

status=0
found=0
for ydir in "$INPUT_ROOT"/*/youtube; do
  [[ -d "$ydir" ]] || continue
  crisis="$(basename "$(dirname "$ydir")")"
  if [[ -n "$CRISES" && ",$CRISES," != *",$crisis,"* ]]; then continue; fi

  phases=""
  for csv in "$ydir"/*.csv; do
    [[ -f "$csv" ]] || continue
    p="$(basename "$csv" .csv)"
    phases="${phases:+$phases,}$p"
  done
  [[ -n "$phases" ]] || { echo "No phase CSVs for $crisis, skipping."; continue; }

  found=1
  echo "=== [$crisis] phases=[$phases] model=$MODEL chunk=$CHUNK_SIZE 4bit=$LOAD_IN_4BIT ==="
  uv run python "$SCRIPT_DIR/run_local.py" --crisis "$crisis" --phases "$phases" "${ARGS[@]}" || status=1
done

[[ $found -eq 1 ]] || { echo "No crises found under $INPUT_ROOT"; exit 2; }
echo
echo "Overall: $([[ $status -eq 0 ]] && echo 'ALL CLEAN' || echo 'SOME PAIRS FAILED (see debug_raw.jsonl in each phase folder)')"
exit $status
