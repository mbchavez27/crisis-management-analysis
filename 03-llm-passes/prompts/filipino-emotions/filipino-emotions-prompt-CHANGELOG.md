# Filipino-emotions prompt change log

The prompt (`prompts/filipino-emotions/filipino-emotions-prompt-v1.0.txt`) is a
self-contained wrapper + Sta. Maria (2010) framework: it is sent to the LLM as
is (after phase-context insertion), with no separate codebook file. A prompt
change alters how the LLM is instructed, never what the framework means.

Every coded record stores the prompt version it was produced with
(`prompt_version` in `filipino_emotions_results.jsonl`), read from the
`# filipino-emotions-prompt vX.Y` header line.

## Runner schema components-3 — 2026-10-14 (no prompt change)

v1.1/v1.2 no longer define `EMOTION_WORDS` or `CLUSTER_CONFLICT`, but the
runner still required them, so screened-in rows came back with
`CLUSTER_CONFLICT = "NA"` (`bad_na` hard failures) and unconstrained chunks
failed `missing_keys`. The runner now derives both: `EMOTION_WORDS` from the
literal-term regexes (`["NONE"]` if none), `CLUSTER_CONFLICT` from
`component_conflict()` (0 when CLUSTER is UNCLASSIFIED); NA when screened
out. Omitted `MULTIPLE_FOCAL` / `NEEDS_CONTEXT` default to 0 (autofix
flag). `--chunk-size` default 5 -> 1. Do not pool with components-2 rows.

Follow-up (same schema; recomputed from saved raw outputs on rerun):
`REGULATION = CONTROLLED` no longer counts toward CLUSTER_CONFLICT. Every
cluster's regulation is "the actor controls the emotion", and the prompt
makes CONTROLLED the catch-all, so it flagged ~75% of pilot rows. New
`CONFLICT_COMPONENTS` column lists the clashing components (e.g.
`BLAME_LOCUS` for C2 rows that blame others); the validator reports counts
per cluster:component. C2 + BLAME_LOCUS=OTHER is kept as a conflict and
reported as a framework-fit finding, not recoded.

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

## v1.2 — 2026-10-14

Fit fix over v1.1 (rules and teaching unchanged — same screen-outs,
TARGET GROUP rule, sarcasm anchors, evidence rule, all 16 inputs and
`What it teaches` lines kept). v1.1's wrapped template was 8179 tokens,
past the L4's 8192 window (7680 after the 512 output reservation), so
every chunk aborted at preflight.

Trims (examples only, documented in the EXAMPLES intro — constrained
decoding still enforces every required key, and the runner autofixes
screened-out rows to NA/{}): `text_head` dropped (runner computes it);
`MULTIPLE_FOCAL`/`NEEDS_CONTEXT` dropped (=0 in all examples);
empty NOTEs and HIGH confidence dropped; English-only `evidence_en`
shown as `"same keys"` (always output full objects); screened-out
examples show id/LANGUAGE/NEG/evidence/NOTE only (all else NA);
ex03/ex12 inputs abridged with `[...]` (all quoted spans kept verbatim).

Runner still loads `components-2`; `PROMPT_FILE` now v1.2. Recodes v1.1
rows (prompt_version differs).

## v1.0 — 2026-10-12

Initial version, converted from `stamaria.txt` with no wording changes to the
framework or instructions:

- MacRoman/CR line endings normalized to UTF-8/LF (`–`/`—`/`…` restored).
- `# filipino-emotions-prompt v1.0` metadata header added so
  `run-filipino-emotions.py` can record `prompt_version` per row.
