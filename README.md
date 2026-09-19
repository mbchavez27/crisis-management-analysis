# Crisis Management Analysis

Text mining approach to analyzing public sentiment and discourse during a multi-phase crisis event — using YouTube comments and video transcripts as primary data sources.

## Project Structure

```
.
├── 01-preprocessing/
│   ├── notebooks/
│   │   ├── youtube-comments-preprocessing.ipynb   # Batch consolidation, cleaning, emoji normalization
│   │   ├── reddit-posts-preprocessing.ipynb       # Reddit thread loading, anonymization, cleaning
│   │   └── video-transcripts-preprocessing.ipynb   # VTT parsing, segment extraction, cleaning
│   └── output/                                     # Cleaned CSVs per crisis phase + transcripts
│   ├── 01-immediate_reaction_24h.csv … 04-institutional_response.csv
│       ├── reddit/
│       │   └── 01-immediate_reaction_24h.csv … 04-institutional_response.csv
│       └── transcripts-{source}.csv
├── 02-analysis/
│   ├── notebook/
│   │   ├── youtube-comments-sentiment-analysis.ipynb              # XLM-RoBERTa sentiment scoring
│   │   ├── youtube-comments-topic-modeling.ipynb                  # BERTopic unsupervised topic discovery
│   │   ├── youtube-comments-sentiment-change-across-phases.ipynb  # Cross-phase sentiment trajectory
│   │   ├── video-transcripts-sentiment-analysis.ipynb             # Transcript sentiment scoring
│   │   └── video-transcripts-topic-modeling.ipynb                 # Transcript topic modeling
│   └── output/
│       ├── scored_{phase}.csv / scored_transcripts-{source}.csv   # Sentiment-labeled outputs
│       └── topics_{phase}.csv / topics_transcripts-{source}.csv   # Topic-labeled outputs
├── 03-insights/
│   └── notebooks/
│       └── summary.ipynb    # Consolidated findings: trajectories, phase rankings, topic shifts
├── 00-data/
│   ├── youtube-comments/
│   │   ├── 01-immediate_reaction_24h/ … 05-aftermath/   # Raw YouTube comment CSVs per crisis phase
│   │   └── {video_id}_comments.csv              # One file per video, placed in its crisis phase
│   ├── reddit-data/
│   │   ├── 00-unrelated/                        # Non-crisis threads & pre-event posts
│   │   ├── 01-the_incident/ … 08-legal_probe/   # Reddit phase folders (mirrors YouTube annotation)
│   │   │   ├── ChikaPH/                         # Subreddit subfolders inside every phase
│   │   │   ├── Philippines/
│   │   │   └── UAAP/
│   │   └── phase-mapping.csv                    # post_id → phase assignment audit trail
│   └── video-transcripts/                       # Raw VTT transcript files
├── outputs/
│   └── figures/          # Generated visualizations
│       ├── reddit-comments-per-phase.png    # Cross-phase comment distribution (Reddit)
│       ├── youtube-comments-per-phase.png   # Cross-phase comment distribution (YouTube)
│       └── *.png                            # Sentiment trajectories, heatmaps, etc.
├── pyproject.toml        # Project metadata & dependencies (uv)
├── uv.lock               # Lockfile
└── main.py               # Entry point (placeholder)
```

## Crisis Phases

The dataset spans 5 chronological phases of a single crisis event, defined by
trigger-based boundaries:

| # | Phase | Trigger Boundary |
|---|-------|-----------------|
| 01 | Immediate Reaction (24h) | Incident → +24 hours exactly |
| 02 | Immediate Reaction (Extended) | +24h → first official confirmation / 72h |
| 03 | Escalation | Phase 2 end → first formal investigation opens |
| 04 | Institutional Response | Formal investigation → first accountability outcome |
| 05 | Aftermath / Long Tail | Formal outcome → pre-registered cutoff |

## Reddit Dataset

Reddit threads are organized to mirror the YouTube comment phase folders. Unlike the
previous structure (`reddit-data/REDDIT/{subreddit_or_keyword}/{scrape_keyword}/`), each
thread now lives under the crisis phase its content belongs to:

- **Phase folders** (`01-the_incident/` … `08-legal_probe/`) match the original YouTube annotation
  semantics — assignment is content-based (thread title + top comments), informed by post
  date, not purely chronological. Reddit uses the legacy 9-phase structure.
- **`00-unrelated/`** holds threads that predate the crisis (before 2026-06-08) or are
  unrelated to it (e.g., general basketball, football, politics threads) so they never
  pollute per-phase corpora.
- **Subreddit subfolders** (`ChikaPH/`, `Philippines/`, `UAAP/`) replicate inside each phase.
  The folder name is read from the CSV `subreddit` column.
- **Deduplication** — duplicate scrapes of the same post (the old structure stored one file
  per scrape keyword) were collapsed to the single copy with the most `comment` rows;
  redundant `.xlsx` exports were dropped. Every unique post is preserved exactly once
  across the tree.
- **`phase-mapping.csv`** records the `post_id → phase` assignment (with date, subreddit,
  comment count, and title) as an audit trail and to re-run/preprocess the split.

## Pipeline

1. **Preprocessing** — Consolidates raw CSVs per phase, cleans text (lowercasing, URL/digit stripping, emoji normalization), and tokenizes into sentences.
2. **Sentiment Analysis** — Classifies text using `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual, 3-class). Uses most-confident-sentence aggregation for long comments.
3. **Topic Modeling** — Unsupervised topic discovery via BERTopic (sentence-transformers embeddings → UMAP → HDBSCAN → c-TF-IDF). Includes manual topic labeling.
4. **Cross-Phase Analysis** — Measures sentiment trajectory across phases, within-phase outrage decay (day 1 vs rest), and phase boundary sentiment shifts.
5. **Summary** — Consolidates all findings into narrative insights with visualizations.

## Setup

```bash
uv sync
```

Requires Python >= 3.11. Key dependencies: `pandas`, `numpy`, `emoji`, `nltk`, `transformers`, `bertopic`, `sentence-transformers`, `spacy`, `matplotlib`, `seaborn`, `scipy`, `torch`.

## License

MIT
