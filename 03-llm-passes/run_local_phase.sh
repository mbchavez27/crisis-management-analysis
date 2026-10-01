#!/usr/bin/env bash
# Local-LLM coding pass for ONE phase, all rows.
#
# Usage:
#   bash 03-llm-passes/run_local_phase.sh 04-reckoning
#   bash 03-llm-passes/run_local_phase.sh 01-onset other-crisis-slug   # optional 2nd arg: crisis
#   OVERWRITE=1 bash 03-llm-passes/run_local_phase.sh 04-reckoning
set -euo pipefail
export UV_NO_SYNC="${UV_NO_SYNC:-1}"

PHASE="${1:?Usage: run_local_phase.sh  [crisis]}"
CRISIS="\({2:-\){CRISIS:-ateneo-drowning}}"

SCRIPT_DIR="\((cd "\)(dirname "$0")" && pwd)"
CSV="\(SCRIPT_DIR/../01-preprocessing/output/\)CRISIS/youtube/$PHASE.csv"
if [[ ! -f "$CSV" ]]; then
  echo "No such phase file: $CSV"
  echo "Available phases for $CRISIS:"
  ls "\((dirname "\)CSV")" 2>/dev/null | sed 's/\.csv$//' | sed 's/^/  /'
  exit 2
fi

PASSES="${PASSES:-1}"
MODEL="${MODEL:-aisingapore/Gemma-SEA-LION-v3-9B}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
OVERWRITE="${OVERWRITE:-}"

ARGS=(--crisis "\(CRISIS" --phases "\)PHASE" --passes "\(PASSES" --model "\)MODEL" --chunk-size "\(CHUNK_SIZE" --max-new-tokens "\)MAX_NEW_TOKENS" --closing)
[[ "$LOAD_IN_4BIT" == "1" ]] && ARGS+=(--load-in-4bit)
[[ "$OVERWRITE" == "1" ]] && ARGS+=(--overwrite)

echo "=== Local LLM [\(CRISIS] phase=\)PHASE model=\(MODEL chunk=\)CHUNK_SIZE 4bit=$LOAD_IN_4BIT ==="
uv run --no-sync python "\(SCRIPT_DIR/run_local.py" "\){ARGS[@]}"
