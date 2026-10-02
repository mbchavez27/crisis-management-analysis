# Coding prompt change log

The coding prompt (`prompts/coding-prompt.txt`) is the wrapper the LLM coder receives
around the codebook. It is versioned separately from the codebook: a prompt change
alters how the LLM is instructed, never what a code means. Code definitions live only
in the codebook, which human and LLM coders share.

Every coded record stores the prompt version it was produced with (`prompt_version`
in `full_codebook_results.jsonl`), read from the `# coding-prompt vX.Y` header line.

## v1.1 — 2026-10-02

Codebook: v2.1 (definitions unchanged).
Prompted by: smoke test on 10 comments from 01-onset (Gemma-SEA-LION-v3-9B-IT, 4-bit):
three rows dropped required keys, one comment quoting a blame-rejecting remark was
coded ACCUSATION / CRITICAL, and every row reported confidence = HIGH.

Prompt text:

- **Added INPUT section.** States that comments arrive as `comment_id | comment text`
  lines under COMMENTS TO CODE, with internal line breaks flattened, and that the id
  must be copied exactly. Previously the input format was never described.
- **Added instruction 6 (evidence first).** The coder quotes and translates the
  supporting words before assigning codes, and codes what the commenter asserts or
  clearly implies rather than the topics mentioned. Restates codebook general rules
  1 and 3; adds no new rule.
- **Added CONSISTENCY CHECKS.** Collects the cross-field rules already in the codebook
  (F2, H2/H3, general rule 8, I4, list-field NONE/NA use, FULL-mode EMERGENT) into one
  checklist at the point of output. These are the failures that output-format
  constraints cannot catch.
- **Defined `confidence`.** HIGH / MEDIUM / LOW were previously undefined.
  `confidence` is a prompt-only field and is not part of the codebook.
- **Moved `evidence` and `evidence_en` to directly after `LANGUAGE`** in the OUTPUT
  template, matching instruction 6. Key set, value formats and placeholder text are
  unchanged.
- **Added a `#` metadata header** carrying the version. It is stripped before sending.

Pipeline (`run_local.py`) changes made alongside this version:

- **Analysis-only codebook sections are omitted from the LLM's copy.** Everything from
  "DERIVED VARIABLES (COMPUTED IN ANALYSIS, NOT CODED)" onward (derived variables,
  mapping to theories, change log) is cut. These sections describe analysis, not
  coding, so human and LLM coders still work from identical code definitions. Saves
  about 660 tokens per request.
- **Generation order follows the template:** id, LANGUAGE, evidence, evidence_en, then
  the codes. The results CSV keeps its original column order.
- **Output is schema-constrained** when coding one comment per call (`--chunk-size 1`):
  every key is required, codes can only take codebook values, the id is pinned to the
  comment, and EMERGENT is forced to "" in FULL mode.
- **`prompt_version` is recorded** in every output record.

Comparability: outputs from v1.0 and v1.1 should not be pooled without checking
agreement on a shared sample (e.g. the 200 hand-coded pilot comments).

## v1.0

Original wrapper used for the pilot passes. Unversioned at the time; records produced
without a `prompt_version` field were made with this version.
