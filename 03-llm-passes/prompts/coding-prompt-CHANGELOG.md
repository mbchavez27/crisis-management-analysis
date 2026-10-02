# Coding prompt change log

The coding prompt (`prompts/coding-prompt.txt`) is the wrapper the LLM coder receives
around the codebook. It is versioned separately from the codebook: a prompt change
alters how the LLM is instructed, never what a code means. Code definitions live only
in the codebook, which human and LLM coders share.

Every coded record stores the prompt version it was produced with (`prompt_version`
in `full_codebook_results.jsonl`), read from the `# coding-prompt vX.Y` header line.

## v1.4 — 2026-10-02

Codebook: v2.1 (definitions unchanged).
Prompted by: v1.3 test on 10 comments from 01-onset (4-bit): 9 of 10 coded, about
10 s per comment (v1.1: about 20 s). The structured format worked: no format errors,
codes traceable to quotes, fewer unsupported VALUES, LANGUAGE no longer TAGLISH on every
row (6 TAGLISH, 3 ENGLISH). Problems:

- STANCE = NEUTRAL and TARGET_OF_COMMENT = NONE on all 9 rows, including
  "false accusation! they were accidentally drowned", which is DEFENDING. As predicted
  in v1.3, the model rarely cites words for whole-comment judgments, so they defaulted.
- One hard failure: a correction retry fixed an over-long quote, but the second answer
  cited CLASS_FRAME without a CLASS_DIRECTION, and only one retry was allowed.
- The run summary counted LANGUAGE review flags as "autofixed" (7/9), which overstated
  the number of automatic corrections.

Changes:

- **STANCE and TARGET_OF_COMMENT are whole-comment fields.** They are no longer cited
  in evidence items; the model gives them as top-level fields after the evidence
  (new WHOLE-COMMENT CODES section, restating D1 and I1, including that a comment
  rejecting blame is DEFENDING and NEEDS_CONTEXT means NEUTRAL). They require no
  supporting quote. The schema adds both fields with their codebook values.
- **Choose-ONE list and examples** in EVIDENCE ITEMS updated accordingly.
- **Up to two correction retries** per rejected comment (was one). The correction note
  now also reminds about CLASS_FRAME/CLASS_DIRECTION and RESPONSE_TARGET/RESPONSE_EVAL.
- **Run summary** reports LANGUAGE review flags separately from deterministic
  autofixes.

## v1.3 — 2026-10-02

Codebook: v2.1 (definitions unchanged).
Prompted by: v1.2 test on 10 comments from 01-onset (4-bit): 0 of 10 coded. Every row
failed the evidence check: the free-text `«quote» -> CODES` format was not followed
(`evidence_format`), bare `NONE` was written as a code, or codes set in the fields were
not cited (`evidence_missing`, mostly STANCE, TARGET_OF_COMMENT, NORM_VIOLATION).
Diagnosis: v1.2 asked the model to keep two parallel records in agreement (a free-text
evidence string and 33 code fields) and enforced it only by rejection afterwards. The
4-bit 9B model could not do this reliably.

Design change:

- **Codes are written only inside evidence items.** Output is now
  `{"id", "evidence": [{"quote", "quote_en", "codes", "other_label"}], "LANGUAGE",
  "OTHER_NOTE", "confidence"}`. The script derives all 33 code fields from the cited
  codes; anything not cited keeps its default (0 / NONE / NA / NEUTRAL), matching
  general rule 2. Every code is tied to quoted words by construction, so there is no
  second record to keep in sync.
- **Decoding is constrained to this structure.** `codes` can only contain the 87 valid
  code tokens (binaries by name, other codes as NAME=VALUE with codebook values;
  defaults are not citable), so format errors and unknown codes cannot occur.
- **Prompt rewritten for the item format:** instruction 6, EVIDENCE ITEMS (replaces
  EVIDENCE FORMAT), CONSISTENCY CHECKS restated in terms of citing, and OUTPUT.
  CASE CONTEXT, CODING REMINDERS and CONFIDENCE are unchanged from v1.2.

Pipeline (`run_local.py`):

- New schema and `expand_structured()`, which builds the flat 39-field record. Rejects
  (then retries once with a correction note) when a choose-ONE code is cited with two
  values (`scalar_conflict`, with a hint to use STANCE=MIXED) or an OTHER has no label.
- The flat `evidence` / `evidence_en` columns are rendered as
  `«quote» -> CODES; …` / `«translation»; …`, so the results CSV is unchanged in shape
  and every code remains traceable. The v1.2 checks still run on this rendering:
  quotes verbatim in the comment, at most 15 words, one translation per quote.
- Cross-field codebook checks (F2 class direction, H2 response target/eval) and
  autofixes are unchanged.
- EMERGENT is set to "" by the script (FULL mode). A PILOT-mode run would need an
  EMERGENT field added back to the output.

Known trade-off: holistic codes (STANCE, TARGET_OF_COMMENT) are now set only when the
model cites words for them. Where it does not, they default to NEUTRAL / NONE. This
follows general rule 2 but may under-code these fields; check them in the pilot
agreement comparison.

## v1.2 — 2026-10-02

Codebook: v2.1 (definitions unchanged; every new reminder restates an existing rule).
Prompted by: review of the v1.1 smoke test (10 comments, 01-onset, 4-bit). Fixed by
v1.1: no template echoes or bad keys, verbatim quotes, translations present, confidence
varies, the "false accusation… accidentally drowned" comment correctly coded ACCIDENT.
Remaining errors:

- LANGUAGE = TAGLISH for all 10, including fully English comments.
- BLAME_TARGETS = UNIVERSITY and COACH on comments that name no one
  ("justice for divine and rene 🥺💔"; "they were murdered… a life for a life").
- VALUES = CHILD_POTENTIAL on 7 of 10, including an emoji comment and a call for
  investigation; EDUCATION_MOBILITY and SOLIDARITY without support.
- "frat hazing ateneo style" coded IDENTITY_FRAME = NATIONALITY and SADNESS.
- Evidence was the whole comment, so codes could not be traced to words.
- A translation added "This is unacceptable," absent from "ano po ateneo".
- All 10 comments were coded differently from the earlier partial run, showing the
  coding is sensitive to prompt wording; accuracy must be measured against human codes.

Prompt text:

- **CASE CONTEXT is background only.** Added: never code from it; every code must come
  from the comment's words (general rule 1). Targets the invented blame targets.
- **Evidence format: quotes linked to codes.** `«exact words» -> CODE, CODE=VALUE; …`,
  at most 15 words per quote, every non-default code cited, nothing unset cited, OTHER
  named in parentheses (as E and H3 require). evidence_en gives one literal translation
  per quote and adds nothing.
- **Order: evidence, evidence_en, LANGUAGE, then codes.** LANGUAGE was the second field
  written, chosen before the model engaged with the text.
- **Added CODING REMINDERS** for BLAME_TARGETS (named or clearly referred-to targets
  only; unnamed wrongdoing → NONE, framing goes in ATTRIBUTION), VALUES (general rule 4),
  IDENTITY_FRAME (the four F3 identities only), emotions (general rules 1 and 9) and
  LANGUAGE (from the comment's own words).
  Note: "unnamed target → NONE" applies general rule 2 ("when in doubt, code NONE") to a
  case C9 does not address. A codebook v2.2 rule may be added if pilot agreement shows
  human coders disagree here.
- **Consistency checks** now include evidence/field agreement.

Pipeline (`run_local.py`) changes made alongside this version:

- **Evidence check (rejects and retries the row).** Each quote must appear verbatim in
  the comment (case- and whitespace-insensitive, @handles as [USER]) and be at most 15
  words; every non-default code (binary = 1, scalar ≠ NONE/NA/NEUTRAL, each list item)
  must be cited; codes cited but not set are rejected; evidence_en must have one
  translation per quote. Runs before the deterministic autofixes.
  Limitation: the check confirms that each code is tied to real words, not that those
  words support the code. Semantic errors remain visible in the evidence column for
  review rather than being blocked.
- **LANGUAGE flag (review only, never overwrites).** `language_check:no_filipino_words`
  when a comment coded TAGLISH/FILIPINO contains no common Filipino or Cebuano function
  words; `language_check:filipino_words_in_english` for the reverse. Recorded in
  `autofix`.
- **Generation order** updated to id, evidence, evidence_en, LANGUAGE, codes.
- **Correction note** on retry now restates the evidence format.
- **Checkpoints carry the prompt version.** On resume, rows coded with a different
  prompt version are recoded instead of being mixed into the results.

Comparability: v1.1 and v1.2 outputs should not be pooled. Evaluate v1.2 against the
200 hand-coded pilot comments before the full run.

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
