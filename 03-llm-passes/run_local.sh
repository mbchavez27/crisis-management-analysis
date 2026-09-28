#!/usr/bin/env bash
# Local-LLM coding passes (transformers, no API key).
# Opt-in: runs exactly the PHASES x PASSES given, one phase x one prompt per pair.
# Full phase CSVs are coded (no sampling); --limit caps rows for smoke tests.
#
# Usage:
#   uv run bash 03-llm-passes/run_local.sh                                  # single smoke pair below
#   PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh     # one phase, one prompt
#   PHASES=04-reckoning PASSES=1,2,3 uv run bash 03-llm-passes/run_local.sh # one phase, all prompts
#   PHASES=01-onset,04-reckoning PASSES=3 uv run bash 03-llm-passes/run_local.sh
#   CRISIS=other-slug PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"
PHASES="${PHASES:-04-reckoning}"
PASSES="${PASSES:-1}"
MODEL="${MODEL:-aisingapore/Gemma-SEA-LION-v3-9B}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ARGS=(--crisis "$CRISIS" --phases "$PHASES" --passes "$PASSES" --model "$MODEL")
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "=== Local LLM [$CRISIS]: phases=[$PHASES] passes=[$PASSES] model=$MODEL ==="
uv run python "$SCRIPT_DIR/run_local.py" "${ARGS[@]}"
