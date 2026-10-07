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
│   ├── pyproject.toml                    # Separate uv env for the local runner (vLLM 0.11.0)
│   ├── run_local.py                      # Local GPU codebook coder (vLLM, schema-constrained)
│   ├── validate_results.py               # Validation + descriptive statistics of coded output
│   ├── make_pilot_sample.py              # Stratified (class-enriched) human-validation sample
│   ├── pilot/{crisis-slug}/              # Pilot coding sheet, comment ids, design record
│   ├── prompts/
│   │   ├── coding-prompt.txt             # LLM wrapper prompt (versioned, currently v1.5)
│   │   ├── coding-prompt-CHANGELOG.md    # Why each prompt version changed
│   │   ├── codebook-v2.1.md              # Codebook shared by human and LLM coders
│   │   └── phase-context.json            # Per-crisis, per-phase CASE CONTEXT text
│   ├── notebooks/                        # LLM passes (topic-label refinement, phase summaries)
│   └── output/{crisis-slug}/
│       └── full-codebook-coding/
│           ├── {phase}/                  # results CSV/JSONL, failure + pending logs, debug_raw
│           └── _validation/{scope}/      # validate_results.py summary tables
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

### Local LLM runner (GPU) — full-codebook coding

`03-llm-passes/run_local.py` codes every YouTube comment against codebook v2.1 with a
local model. It needs **no API key and no `.env`**: only `uv`, a CUDA GPU, and network
to HuggingFace for the first weight download (~18.5 GB for the default
`aisingapore/Gemma-SEA-LION-v3-9B-IT`). It has its **own environment**
(`03-llm-passes/pyproject.toml`), separate from the notebook pipeline above.

How it works, in short:

- The prompt is `prompts/coding-prompt.txt` (version in its `#` header) + the codebook
  (analysis-only sections omitted) + the phase's CASE CONTEXT from `phase-context.json`.
- One comment per call (`--chunk-size 1`). Decoding is constrained to a JSON schema:
  the model cites binary and list codes only inside evidence items (`quote`,
  `quote_en`, `codes`); the script derives the code fields from them. The choose-ONE
  codes (ATTRIBUTION, STANCE, TARGET_OF_COMMENT, CLASS_DIRECTION, RESPONSE_TARGET,
  RESPONSE_EVAL) and LANGUAGE are whole-comment fields. Every cited code, including
  CLASS_FRAME, is traceable to quoted words.
- Rows are checked (quotes found in the comment, codebook cross-field rules). A
  rejected row is retried up to twice with a correction note. If only soft issues
  remain (unverifiable quote, over-long quote, unlabeled OTHER, incomplete class or
  response pair), the row is **kept** and listed in `review_flags`, so long comments
  are not silently lost. Only unusable output (bad JSON, prompt too long) is a hard
  failure.
- Progress is checkpointed per row. Rerunning the same command resumes; rows coded
  with a different prompt version are recoded, never mixed in.

Requirements:

- NVIDIA driver supporting **CUDA 12.8+** (`nvidia-smi` → "CUDA Version"). vLLM 0.11.0
  ships a torch 2.8 / cu128 build.
- Keep `vllm==0.11.0` pinned: it is the release that supports Gemma 2's full 8K context
  **and** still has bitsandbytes 4-bit. Older vLLM caps Gemma 2 at 4,096 tokens (too
  small for the ~5,700-token prompt); newer vLLM removed bitsandbytes and needs CUDA 13.

#### Setup (both machines)

```bash
cd crisis-management-analysis/03-llm-passes
uv sync
uv run python -c "import torch, vllm, transformers; print(torch.__version__, torch.cuda.is_available(), vllm.__version__, transformers.__version__)"
# expect: 2.8.x(+cu128)  True  0.11.0  4.x
```

After `uv sync`, run everything with `uv run --no-sync` so uv never re-resolves the
environment mid-project.

#### GPU profiles and precision

`--gpu-profile` sets the right options for each machine (explicit flags override it).
Every coded row records its `precision`; resuming at a different precision recodes
rows, each phase folder gets `run_info.json`, and `validate_results.py` warns if
phases mix precisions.

| Profile | Weights | KV cache | Context | At once | Precision label | Use |
|---|---|---|---|---|---|---|
| `rtx3060` | nf4 4-bit | bf16 | 7,168 | 2 | `nf4-weights` | Testing only (secondary) |
| `l4-full` | bf16 | bf16 | 8,192 | 4 | `bf16-weights` | Canonical, slow |
| `l4-fp8` | FP8 | FP8 | 8,192 | 12 | `fp8-weights+fp8-kv` | Fast; 8-bit — disclose and validate on the pilot |
| `a100` | bf16 | bf16 | 8,192 | 8 | `bf16-weights` | Canonical, fast |

`l4-fp8` also turns on `--compact-json` (no optional whitespace in answers: fewer
output tokens, same codes).

#### Running the coding pass

Run the setup once per session, then the block for your machine. Each block starts
with a 10-row test; the full run starts only if all 10 rows code cleanly.

**Setup (every session, inside tmux)**

```bash
tmux new -s coding
cd ~/Documents/crisis-management-analysis/03-llm-passes   # adjust the path on the L4 machine
mkdir -p logs && set -o pipefail
PHASES=$(ls ../01-preprocessing/output/ateneo-drowning/youtube/*.csv \
         | xargs -n1 basename | sed 's/\.csv//' | sort | paste -sd,)
BASE="--crisis ateneo-drowning --passes 1 --chunk-size 1 --max-new-tokens 1024 --closing"
```

Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t coding`.

**1. RTX 3060 — 4-bit, testing only (~10.6 s/comment measured)**

```bash
# 700 per phase + class census (~3.5k comments, ~10–11 h)
uv run --no-sync python run_local.py $BASE --phases 01-onset --gpu-profile rtx3060 \
    --limit 10 --overwrite 2>&1 | tee logs/test-rtx3060.log \
&& uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile rtx3060 \
    --sample-per-phase 700 --class-census 2>&1 | tee logs/run-rtx3060-700.log
```

Close Firefox and other GPU apps first (the desktop shares the card).

**2. L4 full precision — canonical, slow**

```bash
# a) 700 per phase + class census (~3.5k comments, ~8–10 h)
uv run --no-sync python run_local.py $BASE --phases 01-onset --gpu-profile l4-full \
    --limit 10 --overwrite 2>&1 | tee logs/test-l4-full.log \
&& uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile l4-full \
    --sample-per-phase 700 --class-census 2>&1 | tee logs/run-l4-full-700.log

# b) all comments (~11.5k, ~22–28 h; rerun the same command to resume across sessions)
uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile l4-full \
    2>&1 | tee -a logs/run-l4-full-all.log
```

**3. L4 FP8 — fast, needs disclosure**

```bash
# a) all comments (~11.5k, ~5–9 h)
uv run --no-sync python run_local.py $BASE --phases 01-onset --gpu-profile l4-fp8 \
    --limit 10 --overwrite 2>&1 | tee logs/test-l4-fp8.log \
&& uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile l4-fp8 \
    2>&1 | tee logs/run-l4-fp8-all.log

# b) ~10k (3,700 per phase + class census, ~4–8 h)
uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile l4-fp8 \
    --sample-per-phase 3700 --class-census 2>&1 | tee logs/run-l4-fp8-10k.log
```

If the FP8 test fails at startup on the KV cache, retry with
`--attention-backend TRITON_ATTN` (or `FLASHINFER` after `uv add flashinfer-python`).

**Notes**

- Times are estimates: hours ≈ comments × seconds per row (from the test log) ÷ 3,600.
  To choose between full precision and FP8, run both 10-row tests and compare speed and
  codes (NEEDS_CONTEXT, STANCE, LANGUAGE, CLASS_FRAME) on the same comments. Each test
  uses `--overwrite`, so read the first test's CSV before running the second.
- Interrupted? Rerun the same full-run command (no `--limit`, no `--overwrite`); it
  resumes with the same sample. Changing profile or sample size recodes rows instead of
  mixing them, so decide before the long run.
- Sampled runs: each phase folder gets `sample_info.json` (design, seed, sizes,
  weights) and rows get `sample_stratum` / `sample_weight`. Use the weights for any
  phase-level percentage — the class census over-represents class talk on purpose.
- Validate afterwards: `uv run --no-sync python validate_results.py`.

#### Cloud VM setup (GCP)

Creates the VM and environment; then use the run blocks above.

```bash
# 0. One-time: request GPU quota in your region (NVIDIA_L4_GPUS, or NVIDIA_A100_GPUS)
# 1. Create the VM (Deep Learning image = driver preinstalled). L4: g2-standard-8; A100: a2-highgpu-1g
gcloud compute instances create coding-vm \
  --zone=us-central1-a --machine-type=g2-standard-8 \
  --image-family=pytorch-2-cuda12-8-ubuntu-22-04-pt \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB

# 2. SSH in and set up
gcloud compute ssh coding-vm --zone=us-central1-a
nvidia-smi    # expect the GPU and CUDA Version >= 12.8
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone git@github.com:mbchavez27/crisis-management-analysis.git
cd crisis-management-analysis/03-llm-passes && uv sync
uv run hf auth login    # only if the weights download is 401 license-gated

# 3. Run (blocks above), validate, then retrieve results and STOP the VM (idle GPUs bill hourly)
exit
gcloud compute scp --recurse coding-vm:~/crisis-management-analysis/03-llm-passes/output ./vm-output --zone=us-central1-a
gcloud compute instances stop coding-vm --zone=us-central1-a
```

Spot VMs save ~40% (`--provisioning-model=SPOT`): safe because per-row checkpoints
resume after a preemption — rerun the full-run command (without `--overwrite`).

#### `run_local.py` options

| Flag | Default | Notes |
|---|---|---|
| `--crisis` | `ateneo-drowning` | Crisis slug |
| `--phases` | required | Comma-separated phase slugs |
| `--passes` | required | `1` (full-codebook pass) |
| `--chunk-size` | `5` | **Always use `1`** — schema-constrained decoding only applies per comment |
| `--max-new-tokens` | `1024` | Maximum output per comment |
| `--min-new-tokens` | `512` | Guaranteed output per comment; each request gets what the context has left, between the two |
| `--max-model-len` | `8192` | Total context. Use `7168` on a 12 GB card |
| `--load-in-4bit` | off | bitsandbytes nf4 + small-GPU memory profile. 12 GB cards only |
| `--model` | `aisingapore/Gemma-SEA-LION-v3-9B-IT` | Any HF instruct causal LM; vet with `--limit 10` first |
| `--limit` | all rows | Cap rows per phase (smoke tests) |
| `--closing` | off | Append a "JSON only" reminder after the comment |
| `--only-ids` | none | Code only these comment_ids (pilot sample); use with `--run-name pilot` |
| `--run-name` | none | Write to `full-codebook-coding-<name>/` (e.g. `pilot`) |
| `--concurrency` | `3` | Comments generated at once. 3060 (4-bit): `2`; L4: `4`; A100: `8`+ |
| `--sample-per-phase` | all | Seeded random sample of N comments per phase |
| `--class-census` | off | With sampling: keep every class-vocabulary comment too; rows get `sample_weight` |
| `--sample-seed` | `20261005` | Same seed = same comments on every machine |
| `--low-memory` | off | Trim unused vLLM reservations (implied by `--load-in-4bit`); use on 24 GB cards |
| `--gpu-profile` | none | `rtx3060`, `l4-full`, `l4-fp8`, `a100` (see GPU profiles) |
| `--fp8-weights` | off | FP8 weights (8-bit); disclose and validate on the pilot |
| `--fp8-kv-cache` | off | FP8 attention cache: ~2× more comments at once |
| `--no-language-rule` | off | Keep the model's LANGUAGE (no A1 word-evidence correction) |
| `--compact-json` | off | No optional whitespace in answers (fewer tokens, same codes) |
| `--attention-backend` | auto | Override vLLM's attention backend if FP8 cache is rejected |
| `--gpu-memory-utilization` | `0.90` | `0.95` on headless server GPUs (L4, A100) |
| `--overwrite` | off | Discard checkpoints. **Only for fresh tests — never on a resume** |

Exit code `0` = every requested phase fully coded; `1` = some rows failed or are
pending. Rerun the same command (without `--overwrite`) to retry pending rows.

Outputs per phase, in `output/{crisis}/full-codebook-coding/{phase}/`:

| File | Contents |
|---|---|
| `full_codebook_results.csv` | One row per comment; all 39 code fields + `autofix` + `review_flags` + `status` |
| `full_codebook_results.jsonl` | Coded records incl. `model` and `prompt_version` |
| `full_codebook_failures.log` | Rows that failed after retries (code these by hand) |
| `full_codebook_pending.log` | Rows not attempted (engine crash / outage) — rerun to resume |
| `debug_raw.jsonl` | Raw model output of every rejected attempt, with the reason |

The `autofix` column records deterministic codebook corrections (e.g. emoji-only →
`LANGUAGE=EMOJI`) and `language_check:*` flags (flagged, never overwritten).
Codebook-rule corrections also appear in `autofix`: `needs_context->0(not_reply)`
(D2: only replies), `uncodable->0(codes_cited)` (I4), and `language->ENGLISH(rule)` /
`language->FILIPINO(rule)` (A1; off with `--no-language-rule`). They are re-applied on
every export, so changing them never requires recoding.
`review_flags` also marks `blame_unnamed(X)` and `class_frame_no_class_word` for review.
`review_flags` lists rows kept after retries: `evidence_unverified`, `evidence_long`,
`evidence_fuzzy`, `other_unlabeled(X)`, `class_direction_missing`,
`class_direction_without_frame(X)`, `response_incomplete(T/E)`. Report results with
and without flagged rows.

#### Validating results and dataset statistics

```bash
cd crisis-management-analysis/03-llm-passes
uv run --no-sync python validate_results.py                                # all phases
uv run --no-sync python validate_results.py --phases 01-onset              # one phase
uv run --no-sync python validate_results.py --phases 01-onset,04-reckoning # several
uv run --no-sync python validate_results.py --no-distributions             # shorter output
uv run --no-sync python validate_results.py --strict                       # exit 1 on WARN too
```

Per phase it prints coverage, failures (by reason), uncoded rate by comment length,
a class audit (CLASS_FRAME rate; class-vocabulary comments that are uncoded or not
class-framed), NEEDS_CONTEXT rate, comment length, videos, code prevalence,
categorical / multi-label distributions, confidence and review flags; for several
phases it adds combined statistics and a cross-phase table. It warns if the selection
mixes prompt versions or models (do not pool those). Outputs:

- `{phase}/full_codebook_results_validation_flags.csv` — one row per flag
  (ERROR = objectively wrong, WARN = likely wrong, INFO = worth a look)
- `_validation/{scope}/phase_overview.csv` — one row per phase (+ `ALL`)
- `_validation/{scope}/stats_long.csv` — phase × variable × category, n and % (for charts)
- `_validation/{scope}/flags_all.csv` — every flag with its phase

`{scope}` is `all` or the selected phases joined with `+`. The checks find rows to
review; accuracy must still be measured against human codes (the 200 hand-coded pilot
comments).

#### Human validation (pilot sample)

The coded output is only usable for the RQs if it agrees with human coders on the
fields the RQs rely on — above all CLASS_FRAME / CLASS_DIRECTION and the emotions.
CLASS_FRAME is rare (~4%), so the pilot sample oversamples class vocabulary:

```bash
cd crisis-management-analysis/03-llm-passes
uv run --no-sync python make_pilot_sample.py            # 150 random + 75 class-vocabulary comments
# -> pilot/ateneo-drowning/pilot_sample.csv      blind coding sheet for human coders
# -> pilot/ateneo-drowning/pilot_comment_ids.txt  ids for --only-ids
# -> pilot/ateneo-drowning/pilot_design.json      seed, strata, counts (for the methods)

# Code the same comments with the LLM (setup block from "Running the coding pass" first):
# Use the SAME --gpu-profile as the main run, so the pilot measures that precision.
uv run --no-sync python run_local.py $BASE --phases "$PHASES" --gpu-profile l4-full \
    --only-ids pilot/ateneo-drowning/pilot_comment_ids.txt --run-name pilot
# -> output/ateneo-drowning/full-codebook-coding-pilot/
```

Sampling is drawn from the preprocessed comments, so comments the model fails on are
included. The random stratum estimates agreement as fields occur; the class stratum is
an oversample and is reported separately or weighted (`weight` column), never pooled
unweighted. Per-field agreement (percent, Cohen's κ, Gwet's AC1) is computed once the
human codes are in.

#### Prompt and codebook versioning

- Code definitions live only in `prompts/codebook-v2.1.md` (shared with human coders).
- `prompts/coding-prompt.txt` changes how the LLM is instructed, never what a code
  means. Bump the `# coding-prompt vX.Y` header and add an entry to
  `coding-prompt-CHANGELOG.md` for every change.
- Outputs from different prompt versions, models or precisions are not pooled; the
  runner recodes stale checkpoints and the validator warns about mixes.

Shared notes: precision policy — run each crisis's **canonical results in full
precision** (GPU VM) and treat 4-bit output as secondary; don't mix precisions within
one crisis's analysis without disclosing it. GPUs below compute capability 7.5 (e.g.
GTX 10-series) are unsupported by current torch wheels — use the GCP VM instead.

#### Local runner troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False`, "NVIDIA driver … too old" | Driver below CUDA 12.8, or a cu130 torch got installed. Keep `vllm==0.11.0`, `rm -rf .venv uv.lock && uv sync` |
| `capping the max length to the sliding window size (4096)` | vLLM too old. Use the pinned 0.11.0 |
| `Unknown quantization method: bitsandbytes` | vLLM too new. Use the pinned 0.11.0 |
| `CUDA out of memory` while loading weights | Missing `--load-in-4bit` on a 12 GB card |
| `KV cache is needed, which is larger than the available KV cache memory` | Add `--max-model-len 7168`; close other GPU apps |
| `Inference engine died` (EngineDeadError, usually OOM) | Remaining rows are marked pending. Free GPU memory, rerun without `--overwrite` |
| Startup: `Prompt template (…) does not fit` | Prompt + `--max-new-tokens` exceed the context. Lower `--max-new-tokens` or raise `--max-model-len` |
| Hard failures `prompt_too_long` | The comment leaves less than `--min-new-tokens` of context (3060 at 7,168: comments over ~650 tokens). Code by hand or run on the L4/A100 (8,192) |
| `rejected (output_truncated …)` | The answer hit the output limit; it is retried with a note asking for fewer, shorter evidence items |
| `unfilled slot '<<…>>'` | Codebook placeholder in `coding-prompt.txt` is not one of `CODEBOOK_SLOTS` |
| `No phase context for {crisis}/{phase}` | Add the phase to `prompts/phase-context.json` |
| Many `rejected (…); retry` lines | Inspect `debug_raw.jsonl`; a model that fails most rows needs a stronger model or a prompt revision |

Dependencies (`pyproject.toml`, pinned in `uv.lock`): `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `bertopic`, `calamancy` + `tl_calamancy_md` (installed by `setup.sh` from HuggingFace, not PyPI), `spacy`, `nltk` (`punkt_tab`, `stopwords`), `pandas`, `emoji`, `matplotlib`, `seaborn`, `papermill`, `google-genai`, `python-dotenv`. Python 3.11 per `.python-version` (uv auto-provisions). The local runner has its own env in `03-llm-passes/pyproject.toml`: `vllm==0.11.0`, `transformers<5`, `bitsandbytes`, `pandas<3`, `tqdm`.

Raw data (`00-data/`, ~12MB) is tracked in git — no separate download needed. First notebook run also pulls HF models (XLM-R, embeddings) into the HF cache.

Troubleshooting:

- `KeyError: 'GEMINI_API_KEY'` → you skipped `cp .env.example .env`.
- `Can't find model 'tl_calamancy_md'` → rerun `bash setup.sh` (step 2).
- `papermill: command not found` → you ran bare `bash run_*.sh`; use `uv run bash ...` or `source .venv/bin/activate` first.
- Slow on CPU-only lab PC → expected for XLM-R + BERTopic; test one phase first (e.g. `uv run python main.py --stage sentiment-youtube`).

## License

MIT
