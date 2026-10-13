# CLA Package — Ateneo Drowning YouTube Analysis

Per-phase topic-labeled comment datasets with sentiment scores, ready for
external review/coding. Generated 2026-10-14.

## Files

| File | Rows |
|---|---|
| `topics_01-onset.csv` | 856 |
| `topics_02-confirmation.csv` | 2,652 |
| `topics_03-spread.csv` | 5,380 |
| `topics_04-reckoning.csv` | 1,411 |
| `topics_05-lingering.csv` | 1,262 |
| `topic-review-sheet.csv` | 45 topics awaiting human validation |

`topics_*.csv` is a superset of the sentiment output — no separate scored
files are included (they contain no additional rows or columns).

## Columns

`video_id, comment_id, parent_comment_id, is_reply, parent_comment, text,
like_count, create_time, is_shouting, sentences, sentiment_label,
sentiment_score, text_clean, topic_id, topic_label, top_terms`

- `text` is cleaned (lowercased, emojis demojized, URLs stripped).
- `is_reply` / `parent_comment` carry reply context; top-level and orphan
  replies have empty `parent_comment`.
- `topic_id = -1` is the noise cluster (unlabeled by design).

## Lineage

`00-data` raw → `01-preprocessing` (cleaning, pseudonymization, reply
resolution) → XLM-R sentiment (`cardiffnlp/twitter-xlm-roberta-base-sentiment`,
most-confident-sentence aggregation) → BERTopic
(`paraphrase-multilingual-MiniLM-L12-v2` embeddings, UMAP+HDBSCAN,
`nr_topics=10`) → LLM topic labels → this package.

## Label provenance

All 45 topic labels were assigned by `gemma-4-26b-a4b-it` with temperature 0
(see `02-analysis/youtube/relabel_topics_gemma.py`). Topic IDs/terms are
stochastic per BERTopic fit — do not compare raw IDs across re-runs.

## Privacy

User handles are SHA-256 pseudonymized (`user_<8hex>`); unknown mentions are
`user_unknown_mention`; raw handles and original text are dropped. A residual
`@`/`u/` scan found 4 hits, all false positives (`@ coach`, `@ admu` used as
prepositions, `yaw@@ mo!` punctuation) — no handle leaks.

## Review sheet

`topic-review-sheet.csv` lists all 45 topics with top terms, sample comments,
and empty `human_label / action / merge_to / notes` columns. Actions:
`confirm` / `rename` / `merge` / `reject`. Validated decisions are applied via
`02-analysis/youtube/youtube-comments-topic-validation.ipynb` (mode=`apply`).
