# Crisis Management Analysis

Crisis Management Analysis using Text Mining Approach — preprocessing, sentiment analysis, and topic modeling of YouTube comments related to crisis events.

## Project Structure

```
.
├── 01-preprocessing/         # Data preprocessing pipeline
│   ├── notebooks/            # Jupyter notebooks (e.g. youtube-comments-preprocessing)
│   └── output/               # Consolidated batch CSV outputs
├── data/
│   ├── batch_1/              # Raw YouTube comment CSVs (batch 1)
│   ├── batch_2/              # Raw YouTube comment CSVs (batch 2)
│   └── batch_3/              # Raw YouTube comment CSVs (batch 3)
├── .venv/                    # Virtual environment (uv-managed)
├── main.py                   # Entry point
├── pyproject.toml            # Project metadata & dependencies
└── uv.lock                   # Lockfile (uv)
```

## Workflow

1. **Data Collection** — YouTube comments collected in batches under `data/batch_N/`.
2. **Preprocessing** — Notebook-driven pipeline in `01-preprocessing/` that consolidates raw CSVs, cleans text, normalizes emoji, and outputs merged datasets.
3. **Analysis** — Sentiment analysis and topic modeling (upcoming).

## Setup

```bash
uv sync
```

Requires Python >= 3.11. Dependencies: `pandas`, `numpy`, `emoji`.

## License

MIT
