# Filipino-emotions prompt change log

The prompt (`prompts/filipino-emotions/filipino-emotions-prompt-v1.0.txt`) is a
self-contained wrapper + Sta. Maria (2010) framework: it is sent to the LLM as
is (after phase-context insertion), with no separate codebook file. A prompt
change alters how the LLM is instructed, never what the framework means.

Every coded record stores the prompt version it was produced with
(`prompt_version` in `filipino_emotions_results.jsonl`), read from the
`# filipino-emotions-prompt vX.Y` header line.

## v1.0 — 2026-10-12

Initial version, converted from `stamaria.txt` with no wording changes to the
framework or instructions:

- MacRoman/CR line endings normalized to UTF-8/LF (`–`/`—`/`…` restored).
- `# filipino-emotions-prompt v1.0` metadata header added so
  `run-filipino-emotions.py` can record `prompt_version` per row.
