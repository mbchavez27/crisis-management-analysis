# Crisis Management Analysis

Text mining approach to analyzing public sentiment and discourse during multi-phase crisis events — using YouTube comments, Reddit threads, and video transcripts as primary data sources.

## Project Structure

```
.
├── 01-preprocessing/
│   ├── notebooks/
│   │   ├── youtube-comments-preprocessing.ipynb   # Batch consolidation, cleaning, emoji normalization
│   │   ├── reddit-posts-preprocessing.ipynb       # Reddit thread loading, anonymization, cleaning
│   │   └── video-transcripts-preprocessing.ipynb   # VTT parsing, segment extraction, cleaning
│   └── output/{crisis-slug}/                      # Cleaned CSVs per crisis
│       ├── youtube/
│       ├── reddit/
│       └── transcripts/
├── 02-analysis/
│   ├── notebook/
│   │   ├── youtube-comments-sentiment-analysis.ipynb
│   │   ├── youtube-comments-topic-modeling.ipynb
│   │   ├── youtube-comments-sentiment-change-across-phases.ipynb
│   │   ├── youtube-comments-topic-validation.ipynb
│   │   ├── video-transcripts-sentiment-analysis.ipynb
│   │   └── video-transcripts-topic-modeling.ipynb
│   └── output/{crisis-slug}/
│       ├── scored_{phase}.csv / scored_transcripts-{source}.csv
│       └── topics_{phase}.csv / topics_transcripts-{source}.csv
├── 04-llm-passes/
│   ├── notebooks/                        # LLM passes (topic-label refinement, phase summaries)
│   └── output/{crisis-slug}/              # LLM pass outputs per crisis
├── 05-summary/
│   └── notebooks/
│       └── summary.ipynb    # Consolidated findings
├── 00-data/
│   ├── CRISIS-PHASES-TEMPLATE.md   # Canonical phase definitions for adding new crises
│   ├── ateneo-drowning/            # First crisis dataset
│   │   ├── youtube-comments/{phase}/
│   │   ├── reddit-data/{phase}/{subreddit}/
│   │   └── video-transcripts/
│   └── {future-crisis}/            # Add new crises here
├── outputs/
│   └── figures/
├── pyproject.toml
├── uv.lock
└── main.py
```

## Multi-Crisis Structure

Each crisis lives under `00-data/{crisis-slug}/` with its own set of YouTube, Reddit, and transcript data. See `00-data/CRISIS-PHASES-TEMPLATE.md` for the canonical phase definitions and instructions for adding new crises.

### Current Crises

| Slug | Description | Videos | Reddit Threads | Transcripts |
|------|-------------|--------|---------------|-------------|
| `ateneo-drowning` | Ateneo basketball team drowning incident (June 2026) | 58 | 261 | 2 |

## Crisis Phases (5 phases, trigger-based)

| # | Phase | Folder Slug | Trigger Boundary |
|---|-------|-------------|-----------------|
| 01 | Onset | `01-onset` | Incident → +24 hours exactly |
| 02 | Confirmation | `02-confirmation` | +24h → first formal investigation opens, OR 72h — whichever first |
| 03 | Spread | `03-spread` | Phase 2 end → first formal accountability act |
| 04 | Reckoning | `04-reckoning` | First accountability act → last accountability act by same party |
| 05 | Lingering | `05-lingering` | End of Phase 4 → pre-registered cutoff |

## Pipeline

1. **Preprocessing** — Consolidates raw CSVs per phase, cleans text, tokenizes into sentences.
2. **Sentiment Analysis** — XLM-RoBERTa multilingual 3-class sentiment scoring.
3. **Topic Modeling** — BERTopic unsupervised topic discovery.
4. **Cross-Phase Analysis** — Sentiment trajectory, within-phase decay, boundary shifts.
5. **Summary** — Consolidated findings with visualizations.

### Running the Pipeline

```bash
# Full pipeline for default crisis (ateneo-drowning)
python main.py --stage pipeline

# Run specific stage
python main.py --stage preprocess
python main.py --stage analysis

# Clean outputs
python main.py --clean

# Run for specific crisis (set CRISIS env var)
CRISIS=ateneo-drowning bash 01-preprocessing/run_youtube.sh
```

## Setup

```bash
uv sync
```

Requires Python >= 3.11. Key dependencies: `pandas`, `numpy`, `emoji`, `nltk`, `transformers`, `bertopic`, `sentence-transformers`, `spacy`, `matplotlib`, `seaborn`, `scipy`, `torch`.

## License

MIT
