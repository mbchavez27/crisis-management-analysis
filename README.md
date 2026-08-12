# Crisis Management Analysis

Text mining approach to analyzing public sentiment and discourse during a multi-phase crisis event — using YouTube comments and video transcripts as primary data sources.

## Project Structure

```
.
├── 01-preprocessing/
│   ├── notebooks/
│   │   ├── youtube-comments-preprocessing.ipynb   # Batch consolidation, cleaning, emoji normalization
│   │   └── video-transcripts-preprocessing.ipynb   # VTT parsing, segment extraction, cleaning
│   └── output/                                     # Cleaned CSVs per crisis phase + transcripts
│       ├── 01-the_incident.csv … 08-legal_probe.csv
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
│   │   ├── 01-the_incident/ … 08-legal_probe/   # Raw YouTube comment CSVs per crisis phase
│   │   └── {video_id}_comments.csv              # One file per video, placed in its crisis phase
│   ├── reddit-data/
│   │   ├── 00-unrelated/                        # Non-crisis threads & pre-event posts
│   │   ├── 01-the_incident/ … 08-legal_probe/   # Mirror of the YouTube phase folders
│   │   │   ├── ChikaPH/                         # Subreddit subfolders inside every phase
│   │   │   ├── Philippines/
│   │   │   └── UAAP/
│   │   └── phase-mapping.csv                    # post_id → phase assignment audit trail
│   └── video-transcripts/                       # Raw VTT transcript files
├── outputs/
│   └── figures/          # Generated visualizations (sentiment trajectories, heatmaps, etc.)
├── pyproject.toml        # Project metadata & dependencies (uv)
├── uv.lock               # Lockfile
└── main.py               # Entry point (placeholder)
```

## Crisis Phases

The dataset spans 9 chronological phases of a single crisis event:

| # | Phase | Description |
|---|-------|-------------|
| 01 | The Incident | Initial event and immediate reactions |
| 02 | Autopsy Request | Calls for investigation / medical inquiry |
| 03 | Public Outcry | Broader public backlash and media coverage |
| 03.1 | Baldwin Apology | Public figure's formal apology |
| 04 | Scholarships Withdrawal | Institutional consequences (scholarship revocation) |
| 05 | Resignation Press Con | Official resignation and press conference |
| 06 | Townhall Podcast | Community dialogue / podcast discussion |
| 07 | UAAP Ban | Sports league sanctions |
| 08 | Legal Probe | Legal / judicial proceedings |

## Reddit Dataset

Reddit threads are organized to mirror the YouTube comment phase folders. Unlike the
previous structure (`reddit-data/REDDIT/{subreddit_or_keyword}/{scrape_keyword}/`), each
thread now lives under the crisis phase its content belongs to:

- **Phase folders** (`01-the_incident/` … `08-legal_probe/`) match the YouTube annotation
  semantics — assignment is content-based (thread title + top comments), informed by post
  date, not purely chronological.
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
