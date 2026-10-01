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
├── 03-llm-passes/
│   ├── prompts/                          # Versioned LLM pass prompts (notebook-loadable)
│   ├── notebooks/                        # LLM passes (topic-label refinement, phase summaries)
│   └── output/{crisis-slug}/              # LLM pass outputs per crisis
├── 04-summary/
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
├── setup.sh                # Fresh-PC setup (uv sync + calamancy model + NLTK)
├── .env.example            # Template for .env (copy, never commit .env)
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

### Running the Pipeline (always via `uv run`)

```bash
# Full pipeline for default crisis (ateneo-drowning)
uv run python main.py --stage pipeline

# Interactive menu
uv run python main.py

# Run specific stage
uv run python main.py --stage preprocess
uv run python main.py --stage analysis

# List all stages
uv run python main.py --list

# Clean outputs
uv run python main.py --clean

# Direct scripts (equivalent, venv-safe without activation)
uv run bash 01-preprocessing/run_youtube.sh

# Run for specific crisis (set CRISIS env var)
CRISIS=ateneo-drowning uv run bash 01-preprocessing/run_youtube.sh

# Or via make
make pipeline
```

## Setup (fresh clone / lab PC)

Prerequisites: `uv 0.10+`, ~25GB free disk (`.venv` ~6GB + HuggingFace cache ~15GB), internet on first run.

```bash
git clone git@github.com:mbchavez27/crisis-management-analysis.git
cd crisis-management-analysis

# 1. Env + models (or: make setup)
bash setup.sh

# 2. API keys (.env is gitignored — never commit it)
cp .env.example .env  # then fill GEMINI_API_KEY / GEMMA_API_KEY

# 3. Run
uv run python main.py --stage pipeline
```

| Env var | Needed for | Where used |
|---|---|---|
| `GEMINI_API_KEY` | topic-label refinement | `02-analysis/*/topic-modeling.ipynb` via `load_dotenv("../../.env")` |
| `GEMMA_API_KEY` | LLM passes | `03-llm-passes/notebooks/` |
| `CRISIS` | multi-crisis select (default `ateneo-drowning`) | all `run_*.sh`, e.g. `CRISIS=... uv run bash ...` |

### Local LLM runner (GPU)

`03-llm-passes/run_local.sh` needs **no API key and no `.env`** — only `uv`, a CUDA GPU, and network to HuggingFace for the first weight download (~18GB for the default 9B model). `setup.sh` / `.env` are for the notebook pipeline, not this runner.

#### Lab PC (8–12 GB GPU, e.g. RTX 3060 — 4-bit)

```bash
# Prereqs: uv (curl -LsSf https://astral.sh/uv/install.sh | sh), working nvidia-smi, ~35GB free disk
git clone git@github.com:mbchavez27/crisis-management-analysis.git
cd crisis-management-analysis

# Standard Setup: syncs the virtual environment exactly to the project lockfile
uv sync

# OPTIONAL: If you need to manually patch the environment for older NVIDIA drivers:
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ==========================================
# PATH A: STANDARD RUNS (Auto-sync enabled)
# Use these if your environment perfectly matches the lockfile.
# ==========================================

# Smoke test: codes 10 rows, pulls the ~18GB weights, validates GPU + output format
LIMIT=10 LOAD_IN_4BIT=1 PASSES=1 uv run bash 03-llm-passes/run_local.sh

# Real run (4-bit nf4, ~6GB VRAM for the 9B default)
PHASES=04-reckoning PASSES=1 LOAD_IN_4BIT=1 uv run bash 03-llm-passes/run_local.sh

# ==========================================
# PATH B: NO-SYNC RUNS (Preserves manual fixes)
# Use these if you applied custom pip installs (like the PyTorch downgrade) 
# to prevent `uv run` from uninstalling them.
# ==========================================

# Smoke test without syncing:
LIMIT=10 LOAD_IN_4BIT=1 PASSES=1 UV_NO_SYNC=1 uv run bash 03-llm-passes/run_local.sh

# Real run without syncing:
PHASES=04-reckoning PASSES=1 LOAD_IN_4BIT=1 UV_NO_SYNC=1 uv run bash 03-llm-passes/run_local.sh
```

GPUs below compute capability 7.5 (e.g. GTX 10-series) are unsupported by current torch wheels — use the GCP VM instead.

#### GCP VM (A100 40 GB — canonical full precision)

```bash
# 0. One-time: request NVIDIA_A100_GPUS quota (1) in your region — approval can take days
# 1. Create the VM (Deep Learning image = driver preinstalled; pick another pytorch image
#    family with: gcloud compute images list --project=deeplearning-platform-release)
gcloud compute instances create coding-vm \
  --zone=us-central1-a --machine-type=a2-highgpu-1g \
  --image-family=pytorch-2-cuda12-8-ubuntu-22-04-pt \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB

# 2. SSH in and set up
gcloud compute ssh coding-vm --zone=us-central1-a
nvidia-smi    # expect 1x A100-SXM4-40GB
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone git@github.com:mbchavez27/crisis-management-analysis.git
cd crisis-management-analysis && uv sync
uv run hf auth login    # only if the weights download is 401 license-gated

# 3. Smoke test, then run inside tmux (an SSH drop must not kill an 8h+ run)
LIMIT=10 PASSES=1 uv run bash 03-llm-passes/run_local.sh
tmux new -s coding
PHASES=04-reckoning PASSES=1 uv run bash 03-llm-passes/run_local.sh 2>&1 | tee reckoning.log
# remaining phases (after reckoning finishes):
# PHASES=01-onset,02-confirmation,03-spread,05-lingering PASSES=1 uv run bash 03-llm-passes/run_local.sh

# 4. Retrieve results, then STOP the VM (an idle A100 bills ~$3.67/hr)
exit
gcloud compute scp --recurse coding-vm:~/crisis-management-analysis/03-llm-passes/output ./vm-output --zone=us-central1-a
gcloud compute instances stop coding-vm --zone=us-central1-a
```

Spot variant saves ~40% (`--provisioning-model=SPOT`): safe because per-chunk checkpoints resume automatically after a preemption — rerun the same command. One model load is reused across all requested phases, so pass every phase in one invocation when possible.

Shared notes: precision policy — run each crisis's **canonical results in full precision** (GPU VM) and treat 4-bit output as secondary — don't mix precisions within one crisis's analysis without disclosing it. Any standard HF causal LM works via `MODEL=<hf-id>`; vet new models with `LIMIT=10` first (weak models fail as content failures, not crashes). Exit code 1 means rows are still pending — rerun the same command to resume (never `OVERWRITE=1` on a resume).

Dependencies (`pyproject.toml`, pinned in `uv.lock`): `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `bertopic`, `calamancy` + `tl_calamancy_md` (installed by `setup.sh` from HuggingFace, not PyPI), `spacy`, `nltk` (`punkt_tab`, `stopwords`), `pandas`, `emoji`, `matplotlib`, `seaborn`, `papermill`, `google-genai`, `python-dotenv`. Python 3.11 per `.python-version` (uv auto-provisions).

Raw data (`00-data/`, ~12MB) is tracked in git — no separate download needed. First notebook run also pulls HF models (XLM-R, embeddings) into the HF cache.

Troubleshooting:

- `KeyError: 'GEMINI_API_KEY'` → you skipped `cp .env.example .env`.
- `Can't find model 'tl_calamancy_md'` → rerun `bash setup.sh` (step 2).
- `papermill: command not found` → you ran bare `bash run_*.sh`; use `uv run bash ...` or `source .venv/bin/activate` first.
- Slow on CPU-only lab PC → expected for XLM-R + BERTopic; test one phase first (e.g. `uv run python main.py --stage sentiment-youtube`).

## License

MIT
