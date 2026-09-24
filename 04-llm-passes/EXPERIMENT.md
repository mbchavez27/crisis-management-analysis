# 04-llm-passes Experiment Log

Batched LLM annotation passes over crisis comments (Gemma 4 31B via Gemini API).
Each notebook packs `batch_size` comments as `id,comment` CSV per request and
parses back a JSON array. Architecture adapted from the Translation Pipeline v2
(`ThreadPoolExecutor` + `as_completed`, checkpointing, dedicated retry pass).

## Locked experiment spec

| Decision | Value |
|---|---|
| Data | `youtube_backup/02-immediate_reaction_extended.csv`, seeded sample n=2000 (`sample_seed = 42`) |
| Text | Full, no truncation (`max_comment_chars = None`) |
| IDs | Synthesized `{stem}-{idx:05d}`; source CSVs have no `id` column |
| Batch / workers | 25 / 2, shared token-bucket pacing at 12 RPM target |
| Passes | grief-outrage, basic-emotions (7 fixed categories), action-demanded |
| Model | `gemma-4-31b-it`, temperature 0.0, seed 42, thinking minimal |
| Output contract | `response_schema` typed list-of-objects + id-set-equality guard |
| Outputs | Committed under `output/_experiments/{pass}/` + `raw/` JSONL + `run_log.json` |
| Validation bar | 100% id coverage, schema checks, ~15-row manual spot-check per pass |

## Deferred production decisions

| Decision | Trigger criterion | Status |
|---|---|---|
| Batch 25 vs 50 | 0 batch failures on the 2k sample → 50, else stay 25 | pending evidence |
| Production workers | Measured sec/call + AI Studio RPM/TPM quota | pending evidence |
| Keep Gemma vs Gemini Flash fallback | Spot-check quality, esp. prompt-2 category discipline | pending evidence |
| Pipeline wiring (default composite vs opt-in) | First full-run wall time | pending evidence |
| Human validation sample size | Agreement rate on experiment spot-check | pending evidence |

## Results

_Blocked: no `GEMMA_API_KEY` available yet — run `bash 04-llm-passes/run_experiment.sh`
once a key is in `.env`, then fill in parse success rate, id coverage, failure
tallies, sec/call, and spot-check notes here._
