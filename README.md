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
├── data/
│   ├── 01-the_incident/ … 08-legal_probe/    # Raw YouTube comment CSVs per crisis phase
│   └── transcripts-{source}/                 # Raw VTT transcript files
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
