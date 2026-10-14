# Filipino-emotions prompt change log

The prompt (`prompts/filipino-emotions/filipino-emotions-prompt-v1.0.txt`) is a
self-contained wrapper + Sta. Maria (2010) framework: it is sent to the LLM as
is (after phase-context insertion), with no separate codebook file. A prompt
change alters how the LLM is instructed, never what the framework means.

Every coded record stores the prompt version it was produced with
(`prompt_version` in `filipino_emotions_results.jsonl`), read from the
`# filipino-emotions-prompt vX.Y` header line.

## v1.1 — 2026-10-14

Prof revision of v1.0 (sourced from `prompt_stamaria.txt`, deleted after
promotion so v1.1 is the single source of truth). Targets hallucinations
found in the 100-row v1.0 run:

- Grief-only inconsistency (rows 1, 62, 93, 94 coded NEG=1 vs rows 16, 30,
  67 coded NEG=0): STEP 1 screen-out list made explicit — grief, sadness,
  sympathy, prayers, condolences with no one blamed are NEG=0 — plus new
  Example 13 (grief-only) as anchor.
- Defending mis-coded as blaming (rows 0, 39): TARGET rule added — a
  commenter defending the organization and attacking its critics has
  TARGET=GROUP — plus Example 11 (anger at accusers) and Example 14
  (defending the coach with no anger toward anyone is NEG=0).
- Sarcasm missed (rows 55, 56, 65): SARCASM section kept with mocking
  examples; Example 4 (mild/question criticism is still MORAL_VIOLATION)
  and Example 5 (bare `team building?` is SARCASM=1).
- Guardrail: new THE EVIDENCE RULE — every component starts at its default
  and changes only with a quotable span; most comments correctly keep
  NONE_EXPRESSED / NOT_DETERMINABLE.

Output shape change: `evidence` / `evidence_en` are now objects keyed by
component (`{FOCAL_EVENT, BLAME_LOCUS, CONTROL_STRATEGY, ACTION_TENDENCY,
REGULATION, RELATIONSHIP_LEVEL, SARCASM}`), `{}` when screened out — not
free-text strings. Runner schema `components-1` -> `components-2`;
`PROMPT_FILE` now loads v1.1. Do not pool v1.0 and v1.1 outputs.

## v1.0 — 2026-10-12

Initial version, converted from `stamaria.txt` with no wording changes to the
framework or instructions:

- MacRoman/CR line endings normalized to UTF-8/LF (`–`/`—`/`…` restored).
- `# filipino-emotions-prompt v1.0` metadata header added so
  `run-filipino-emotions.py` can record `prompt_version` per row.
