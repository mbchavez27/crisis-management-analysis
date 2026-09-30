#!/usr/bin/env bash
# Local-LLM coding pass (transformers, no API key): v2 full codebook only.
# Pass key 1 = full-codebook coding; legacy notebooks/prompts are under legacy/.
# Opt-in: runs exactly the PHASES x PASSES given, one phase x one prompt per pair.
# Full phase CSVs are coded (no sampling); --limit caps rows for smoke tests.
#
# Usage:
#   uv run bash 03-llm-passes/run_local.sh                                  # single smoke pair below
#   PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh     # one phase, codebook pass
#   PHASES=01-onset,04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh
#   CRISIS=other-slug PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh
#   LOAD_IN_4BIT=1 PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh   # <=12GB GPU
#   MODEL=Qwen/Qwen2.5-7B-Instruct PHASES=... uv run bash 03-llm-passes/run_local.sh     # other model
set -euo pipefail

CRISIS="${CRISIS:-ateneo-drowning}"
PHASES="${PHASES:-04-reckoning}"
PASSES="${PASSES:-1}"
MODEL="${MODEL:-aisingapore/Gemma-SEA-LION-v3-9B}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ARGS=(--crisis "$CRISIS" --phases "$PHASES" --passes "$PASSES" --model "$MODEL")
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  ARGS+=(--load-in-4bit)
fi

echo "=== Local LLM [$CRISIS]: phases=[$PHASES] passes=[$PASSES] model=$MODEL 4bit=${LOAD_IN_4BIT:-0} ==="
uv run python "$SCRIPT_DIR/run_local.py" "${ARGS[@]}"
