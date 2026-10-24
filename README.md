# Crisis Management Analysis

A text-mining study of how the Filipino public talked about a crisis as it unfolded: sentiment,
emotions, class framing, values and demanded actions in YouTube comments, across five crisis
phases. The first (and so far only) case is the June 2026 Ateneo basketball team drowning
incident.

> **Status: work in progress.** All committed outputs are for `ateneo-drowning`. The findings in
> stage 04 cover YouTube comments only. Reddit threads and video transcripts are preprocessed and
> scored but not used in the findings.

## Pipeline at a glance

Every stage is a folder with its own uv environment. Each stage reads the previous stage's
`output/`, and **every output is committed**, so you can start at any stage without re-running
the ones before it.

| Stage | What it does | Needs | Runs on a fresh clone? |
|---|---|---|---|
| `01-preprocessing` | Merges raw comment, Reddit and transcript CSVs per phase; cleans text, normalizes emoji | CPU | Yes |
| `02-analysis` | XLM-RoBERTa sentiment, BERTopic topics (labels from Gemini), sentiment change across phases | CPU works (slow); GPU optional; `GEMINI_API_KEY` for topic labels | Yes |
| `03-llm-passes` | Codes every comment with a local LLM (class-emotions codebook, Filipino emotions) | CUDA GPU (12 GB+), CUDA 12.8 driver | Linux/Windows GPU only. See [`03-llm-passes/README.md`](03-llm-passes/README.md) |
| `04-statistical-analysis` | Joins the LLM codes with sentiment and topics; RQ1–RQ3 tables and figures | CPU | Yes |
| `05-summary` | Cross-phase sentiment trajectory, topic heatmaps, boundary shifts | CPU | Yes |

## Quick start: reproduce the findings (CPU, no API keys, ~5 min)

This regenerates the stage 04 and 05 figures from the committed outputs. It works on macOS,
Linux and Windows (Git Bash or WSL).

```bash
# Prerequisite: uv (https://docs.astral.sh/uv/getting-started/installation/)
git clone https://github.com/mbchavez27/crisis-management-analysis.git
cd crisis-management-analysis

# Stage 04 findings: RQ1 sentiment and emotions, RQ2 class/values/actions, RQ3 endurance
uv sync --project 04-statistical-analysis
uv run --project 04-statistical-analysis --with jupyterlab \
    jupyter lab 04-statistical-analysis/notebooks/03-findings.ipynb
#   → Run All. Tables and PNGs land in 04-statistical-analysis/output/ateneo-drowning/findings/

# Stage 05 summary figures (runs the notebook headless)
uv sync --project 05-summary
bash 05-summary/run_summary.sh
#   → outputs/figures/05-summary/
```

Re-running gives the committed numbers back (the only differences are floating-point noise
around 1e-15), so `git diff` on the outputs is a quick check that your setup matches.

## Full setup (to re-run stages 01–02)

Prerequisites: [uv](https://docs.astral.sh/uv/) 0.10+, about 25 GB of free disk
(`02-analysis/.venv` ~6 GB plus the Hugging Face model cache), and internet on the first run.
uv installs Python 3.11 by itself.

```bash
bash setup.sh          # or: make setup
cp .env.example .env   # then fill in GEMINI_API_KEY (only needed for 02 topic labels)
```

`setup.sh` syncs the environments for stages 01, 02, 04 and 05, installs the Tagalog spaCy model
(`tl_calamancy_md`) into stage 02, and downloads NLTK data. Stage 03 is set up separately, see its
README.

**Platforms.** On Linux and Windows, stage 02 installs the CUDA 12.6 build of torch. On macOS it
installs the regular PyPI build, and the sentiment notebooks run on CPU. Stage 03 needs an NVIDIA
GPU.

## Running stages

`main.py` and `make` call each stage's shell scripts. Each script picks its own stage environment,
so nothing needs activating. `main.py` uses only the standard library.

```bash
python3 main.py --list                 # all stages
python3 main.py                        # interactive menu
python3 main.py --stage preprocess     # 01: YouTube, Reddit, transcripts
python3 main.py --stage analysis       # 02: sentiment, topics, topic validation
python3 main.py --stage insights       # 02 sentiment change + 05 summary
python3 main.py --stage sentiment-youtube   # a single step

make preprocess                        # same stages through make
CRISIS=ateneo-drowning bash 01-preprocessing/run_youtube.sh   # a script directly
```

Things to know before running stage 02:

- `topic-validation` is **interactive**. It writes a review sheet, waits while you fill in
  `human_label` / `action` in the CSV, then applies your decisions.
- Topic modeling sends topic keywords to Gemini to name the topics, so it needs `GEMINI_API_KEY`.
- On CPU, XLM-R and BERTopic are slow. Try one step first (`--stage sentiment-youtube`).

Stage 04 has no run script. Open its notebooks in Jupyter with the stage 04 environment (kernel:
`04-statistical-analysis/.venv`), as in the quick start. Run them in order: `01-merge-and-prep`
writes `analysis_df.csv` for `02-statistical-analysis`. `03-findings` reads stages 02 and 03 directly
and runs on its own.

Stage 03 runs on a GPU machine. `make llm-local` runs a 10-row class-emotions test and
`make llm-validate` validates the results. Full commands, GPU profiles, cloud VM setup and
validation are in [`03-llm-passes/README.md`](03-llm-passes/README.md).

## Data and crisis phases

Raw data lives in `00-data/{crisis-slug}/` (~11 MB, committed): YouTube comments, Reddit threads
and video transcripts. The comments are split into folders by phase.

| Slug | Event | Videos | Reddit threads | Transcripts |
|---|---|---|---|---|
| `ateneo-drowning` | Ateneo basketball team drowning (June 2026) | 58 | 261 | 2 |

Phases are trigger-based (the full definitions are in `00-data/CRISIS-PHASES-TEMPLATE.md`):

| # | Phase | Folder | Boundary |
|---|---|---|---|
| 01 | Onset | `01-onset` | Incident → +24 hours |
| 02 | Confirmation | `02-confirmation` | +24 h → first formal investigation, or 72 h, whichever comes first |
| 03 | Spread | `03-spread` | End of phase 2 → first formal accountability act |
| 04 | Reckoning | `04-reckoning` | First → last accountability act by the same party |
| 05 | Lingering | `05-lingering` | End of phase 4 → pre-registered cutoff |

**Using your own data:** see [`00-data/README.md`](00-data/README.md). It covers the dataset
format (folders, columns, timestamps), a format checker (`python3 00-data/check_dataset.py <slug>`),
and the few per-crisis edits each stage needs.

## Repository layout

```
00-data/{crisis}/                      raw YouTube comments, Reddit threads, transcripts
00-data/README.md, check_dataset.py   dataset format guide and checker
01-preprocessing/   notebooks/, run_*.sh, output/{crisis}/{youtube,reddit,transcripts}/
02-analysis/        youtube/, transcripts/ (notebooks + run_*.sh), output/{crisis}/
03-llm-passes/      scripts/, prompts/, output/{crisis}/   (own README)
04-statistical-analysis/  notebooks/01-merge-and-prep, 02-statistical-analysis, 03-findings
                          output/{crisis}/findings/
05-summary/         notebooks/summary.ipynb, run_summary.sh
outputs/figures/    figures from 01, 02 and 05
main.py, Makefile   stage runners
setup.sh            one-shot setup for stages 01, 02, 04, 05
```

Each stage folder has its own `pyproject.toml`, `uv.lock` and `.python-version` (3.11). There is
no root environment.

## Reproducibility

- Every stage environment is locked (`uv.lock`) and pinned to Python 3.11.
- Stage 02 (sentiment models and BERTopic embeddings) runs on CUDA or CPU, never Apple MPS. CUDA and CPU can still differ in the
  last digits, so make reported results on a single machine.
- Stage 03 records the prompt version, model and precision in every coded row and never mixes
  them on resume. Canonical results use full precision; see its README.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `KeyError: 'GEMINI_API_KEY'` | `cp .env.example .env` and fill in the key |
| `404 … gemini-… no longer available` | Set `MODEL` in the topic-modeling notebook to a current Gemini model (default `gemini-3.5-flash-lite`) |
| `Can't find model 'tl_calamancy_md'` | Re-run `bash setup.sh`. A `uv sync` of 02 removes the model, and step 2 reinstalls it |
| `papermill: command not found` or a missing module | That stage isn't synced: `uv sync --project <stage>` |
| 02 is very slow | Expected on CPU. Test one phase, or use a CUDA machine |
| Stage 03 problems | See the troubleshooting table in [`03-llm-passes/README.md`](03-llm-passes/README.md) |

## License

MIT
