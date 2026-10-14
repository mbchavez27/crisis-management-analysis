r"""Local-LLM runner for the filipino-emotions coding pass (Sta. Maria 2010 framework).

Codes full phase CSVs against the self-contained filipino-emotions-prompt-v1.1
(wrapper + framework in one file; only the per-phase case context is inserted).
Supersedes the removed notebooks/full-codebook-coding.ipynb micro-sample path.

Opt-in execution: --phases and --passes are both required, and the script
runs exactly their cartesian product (one phase x one prompt per run pair).

Run from 03-llm-passes/ (where pyproject.toml lives, so uv resolves the env).

Example:
    uv run python scripts/filipino-emotions/run-filipino-emotions.py --phases 04-reckoning --passes 1
    uv run python scripts/filipino-emotions/run-filipino-emotions.py --phases 04-reckoning --passes 1 --limit 10
    uv run python scripts/filipino-emotions/run-filipino-emotions.py --phases 04-reckoning --passes 1 --load-in-4bit

    # All phases, ONE model load (comma-separated):
    PHASES=$(ls ../01-preprocessing/output/ateneo-drowning/youtube/*.csv \
             | xargs -n1 basename | sed 's/\.csv//' | sort | paste -sd,)
    uv run --no-sync python scripts/filipino-emotions/run-filipino-emotions.py --crisis ateneo-drowning \
        --phases "$PHASES" --passes 1 --chunk-size 1 --load-in-4bit --closing

--load-in-4bit: nf4 quantization (bitsandbytes, applied by vLLM at load time)
for <=12GB GPUs. Required for the 9B model on a 12GB card. Full precision is the
default so canonical runs (GPU VM) stay byte-identical across machines.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
STAGE_ROOT = HERE.parent.parent
PROMPTS_DIR = STAGE_ROOT / "prompts"
PASS_PROMPTS_DIR = PROMPTS_DIR / "filipino-emotions"
OUTPUT_ROOT = STAGE_ROOT / "output"

# Instruction-tuned SEA-LION v3 (the plain "-9B" checkpoint is the base model).
DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT"
# Total context (prompt + output). Gemma 2's native limit is 8192.
MAX_MODEL_LEN = 8192
SKIP_SUMMARY_COLS = {"row_id", "comment_id", "video_id", "text", "id", "_row_id", "row_ids", "status", "sample_weight"}

# Outage-resilient retry: exponential backoff, requeue, circuit breaker.
# Values kept identical to the former notebook inference cell.
TRANSIENT_TRIES = 5       # 5 attempts => transient sleeps ~10/20/40/80s
TRANSIENT_BASE_S = 10
BACKOFF_CAP_S = 300
RATE_LIMIT_SLEEP = (60, 90)
COOLDOWN_S = 60           # pause before a requeued chunk is attempted again
MAX_REVISITS = 3          # requeues per chunk before it is marked pending
BREAKER_TRIP = 5          # consecutive outage-exhausted chunks before abort
MAX_CORRECTIONS = 2       # corrective retries per single-row content rejection

# Optional reminder after the comments. OFF by default: on emoji-only comments the
# model copied this sentence into `evidence`. The forced `{"id":"..."` prefill is
# what keeps output JSON-only. Enable with --closing if echo loops come back.
CLOSING = (
    "\n\nRespond with the JSON object(s) only, one per line, as specified in OUTPUT."
)

# Substrings that only appear if the model copied the output template or our own
# instructions into a field instead of coding the comment.
PLACEHOLDER_MARKERS = (
    "short quote(s) supporting",
    "english translation of the evidence",
    "high|medium|low",
    "en|fil|taglish|other",
    "one json object per comment",
    "output only json",
    "respond with the json",
    "now output exactly",
    "[id | comment text]",
)

# Bumped whenever the output fields change. Checkpoints written under another schema
# are recoded, never mixed in (prompt_version alone cannot tell: v1.0 is reused).
SCHEMA_VERSION = "components-2"

# Evidence components keyed in the v1.1 object-form evidence (prompt_stamaria revision).
# Screened-out rows use {} for both evidence objects.
EVIDENCE_KEYS = ("FOCAL_EVENT", "BLAME_LOCUS", "CONTROL_STRATEGY", "ACTION_TENDENCY",
                 "REGULATION", "RELATIONSHIP_LEVEL", "SARCASM")

# --- Output schema of filipino-emotions-prompt v1.1 ---------------------------------
# Step 1 screens (NEG_SOCIAL_EMOTION); if 0, every other coded field is "NA" and both
# evidence objects are {}. Step 2 codes the components; Step 3 assigns the cluster.
BINARY_KEYS = ("NEG_SOCIAL_EMOTION", "SARCASM", "NEEDS_CONTEXT")  # always 0 / 1
TERNARY_KEYS = ("MULTIPLE_FOCAL", "CLUSTER_CONFLICT")  # 0 / 1, or "NA" when screened out
COMPONENT_KEYS = (
    "TARGET", "FOCAL_EVENT", "BLAME_LOCUS", "CONTROL_STRATEGY", "ACTION_TENDENCY",
    "REGULATION", "RELATIONSHIP_LEVEL", "CLUSTER",
)
SCALAR_ENUMS: dict[str, frozenset[str]] = {
    "LANGUAGE": frozenset({"EN", "FIL", "TAGLISH", "OTHER"}),
    "TARGET": frozenset({"PERSON", "ORGANIZATION", "GROUP", "SITUATION_OBJECT", "SELF",
                         "UNCLEAR", "NA"}),
    "FOCAL_EVENT": frozenset({"MORAL_VIOLATION", "UNMET_EXPECTATION", "HINDRANCE", "UNCLEAR", "NA"}),
    "BLAME_LOCUS": frozenset({"OTHER", "SELF", "NOT_EXPRESSED", "NA"}),
    "CONTROL_STRATEGY": frozenset({"IGNORE", "REFRAME", "NONE_EXPRESSED", "NA"}),
    "ACTION_TENDENCY": frozenset({"LASTING_WITHDRAWAL", "TEMPORARY_DISTANCE",
                                  "RELATIONSHIP_MAINTAINED", "NONE_EXPRESSED", "NA"}),
    "REGULATION": frozenset({"INTENSE_DISTRESS", "CONTROLLED", "NONVERBAL_ONLY",
                             "NOT_DETERMINABLE", "NA"}),
    "RELATIONSHIP_LEVEL": frozenset({"KAPWA", "HINDI_IBANG_TAO", "IBANG_TAO",
                                     "NOT_DETERMINABLE", "NA"}),
    "CLUSTER": frozenset({"C1", "C2", "C3", "UNCLASSIFIED", "NA"}),
    "confidence": frozenset({"HIGH", "MEDIUM", "LOW"}),
}
EMOTION_TERMS = (
    "ASAR", "BUWISIT", "GALIT", "INGGIT", "INIS", "MUHI", "NGITNGIT", "PIKON", "POOT",
    "SAMA_NG_LOOB", "SELOS", "SUKLAM", "SUYA", "TAMPO", "YAMOT",
)
EMOTION_WORD_VALUES = frozenset(EMOTION_TERMS) | {"NONE", "NA"}  # NONE: no term; NA: screened out

FREE_TEXT_KEYS = ("NOTE",)
EVIDENCE_DICT_KEYS = ("evidence", "evidence_en")


def build_record_schema(comment_id: str) -> dict[str, Any]:
    """JSON schema constraining decoding for one comment (key order = prompt's OUTPUT order).

    Every key is present and every coded value comes from the codebook, so the output cannot
    be malformed. Cross-field rules (screened-out rows are all NA; screened-in rows have no
    NA) cannot be expressed here and are enforced in validate_record().
    """
    def enum(key: str) -> dict[str, Any]:
        return {"type": "string", "enum": sorted(SCALAR_ENUMS[key])}

    props: dict[str, Any] = {
        "id": {"type": "string", "enum": [str(comment_id)]},
        "LANGUAGE": enum("LANGUAGE"),
        "NEG_SOCIAL_EMOTION": {"enum": [0, 1]},
        "EMOTION_WORDS": {"type": "array", "minItems": 1,
                          "items": {"type": "string", "enum": sorted(EMOTION_WORD_VALUES)}},
        "TARGET": enum("TARGET"),
        "FOCAL_EVENT": enum("FOCAL_EVENT"),
        "MULTIPLE_FOCAL": {"enum": [0, 1, "NA"]},
        "BLAME_LOCUS": enum("BLAME_LOCUS"),
        "CONTROL_STRATEGY": enum("CONTROL_STRATEGY"),
        "ACTION_TENDENCY": enum("ACTION_TENDENCY"),
        "REGULATION": enum("REGULATION"),
        "RELATIONSHIP_LEVEL": enum("RELATIONSHIP_LEVEL"),
        "CLUSTER": enum("CLUSTER"),
        "CLUSTER_CONFLICT": {"enum": [0, 1, "NA"]},
        "SARCASM": {"enum": [0, 1]},
        "NEEDS_CONTEXT": {"enum": [0, 1]},
        "evidence": {"type": "object",
                     "properties": {k: {"type": "string"} for k in EVIDENCE_KEYS},
                     "additionalProperties": False},
        "evidence_en": {"type": "object",
                        "properties": {k: {"type": "string"} for k in EVIDENCE_KEYS},
                        "additionalProperties": False},
        "NOTE": {"type": "string"},
        "confidence": enum("confidence"),
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_SHORTCODE = re.compile(r":[a-z0-9_+\-]+:", re.I)
_UNI_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_HANDLE = re.compile(r"@[\w.\-]+")
_LEFTOVER_SLOT = re.compile(r"<<[^<>\n]*>>")


def is_rate_limited(e: Exception) -> bool:
    msg = repr(e)
    return ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg)


def is_transient(e: Exception) -> bool:
    msg = repr(e)
    return any(
        k in msg
        for k in (
            "500",
            "INTERNAL",
            "503",
            "UNAVAILABLE",
            "HIGH DEMAND",
            "Timeout",
            "TIMEOUT",
            "TimeoutError",
            "timed out",
        )
    )


class TruncatedOutput(ValueError):
    """The answer hit the output-token limit and was cut off (finish_reason='length').

    Parsing a cut-off JSON object would report misleading errors (a fragment
    looks like a record with missing keys), so it is flagged as truncation and
    retried with a note asking for fewer, shorter evidence items.
    """

    def __init__(self, text: str) -> None:
        super().__init__("output_truncated: the answer exceeded the output budget and was cut off")
        self.text = text


def is_prompt_too_long(e: Exception) -> bool:
    """A comment that cannot fit the context window; retrying cannot help."""
    return "prompt_too_long" in repr(e)


def is_engine_dead(e: Exception) -> bool:
    """The local inference engine crashed (e.g. GPU out of memory); nothing will recover it."""
    return type(e).__name__ == "EngineDeadError" or "EngineDeadError" in repr(e)


def generate_with_retry(
    generator: Any, prompt: str, tries: int = TRANSIENT_TRIES, **kw: Any
) -> str:
    """Generate one chunk with exponential backoff for outages (mirrors the notebook)."""
    for attempt in range(tries):
        try:
            return generator.generate(prompt, **kw)
        except Exception as e:
            if attempt == tries - 1 or is_engine_dead(e) or is_prompt_too_long(e):
                raise
            if is_rate_limited(e):
                sleep_time = random.uniform(*RATE_LIMIT_SLEEP)
            elif is_transient(e):
                sleep_time = min(TRANSIENT_BASE_S * (2 ** attempt) + random.uniform(0, 5), BACKOFF_CAP_S)
            else:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [Attempt {attempt+1} failed: {repr(e)}] Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    raise RuntimeError("generate_with_retry: no attempts made")


def generate_batch_with_retry(
    generator: Any, jobs: list[tuple[str, str, Optional[dict[str, Any]]]],
    tries: int = TRANSIENT_TRIES,
) -> list[Any]:
    """Batch version of generate_with_retry: returns one text or Exception per job."""
    for attempt in range(tries):
        try:
            return generator.generate_batch(jobs)
        except Exception as e:
            if attempt == tries - 1 or is_engine_dead(e):
                return [e] * len(jobs)
            if is_transient(e) or is_rate_limited(e):
                sleep_time = min(TRANSIENT_BASE_S * (2 ** attempt) + random.uniform(0, 5), BACKOFF_CAP_S)
            else:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [Batch attempt {attempt+1} failed: {repr(e)[:200]}] Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    return [RuntimeError("generate_batch_with_retry: no attempts made")] * len(jobs)


def append_pending_rows(path: Path, chunk: pd.DataFrame, reason: str, error: str) -> None:
    """Append chunk rows to the pending (transient-outage) checkpoint."""
    with open(path, "a", encoding="utf-8") as f:
        for _, row in chunk.iterrows():
            ent = {
                "row_id": int(row["row_id"]),
                "comment_id": row.get("comment_id"),
                "reason": reason,
                "error": error,
                "text_length": len(str(row["text"])),
            }
            f.write(json.dumps({"kind": "pending_transient", **ent}, ensure_ascii=False) + "\n")


def dump_raw(path: Path, row_ids: list[int], err: str, raw: str) -> None:
    """Keep the raw model output for every failed chunk so failures can be inspected."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"row_ids": row_ids, "err": err, "raw": raw}, ensure_ascii=False) + "\n")


REQUIRED_KEYS = frozenset(
    {"id", "LANGUAGE", "NEG_SOCIAL_EMOTION", "EMOTION_WORDS", *COMPONENT_KEYS, *TERNARY_KEYS,
     "SARCASM", "NEEDS_CONTEXT", "evidence", "evidence_en", "NOTE", "confidence"}
)

COLUMN_ORDER = [
    "row_id",
    "comment_id",
    "video_id",
    "text",
    "id",
    "LANGUAGE",
    "NEG_SOCIAL_EMOTION",
    "EMOTION_WORDS",
    "TARGET",
    "FOCAL_EVENT",
    "MULTIPLE_FOCAL",
    "BLAME_LOCUS",
    "CONTROL_STRATEGY",
    "ACTION_TENDENCY",
    "REGULATION",
    "RELATIONSHIP_LEVEL",
    "CLUSTER",
    "CLUSTER_CONFLICT",
    "SARCASM",
    "NEEDS_CONTEXT",
    "evidence",
    "evidence_en",
    "NOTE",
    "confidence",
    "autofix",
    "review_flags",
    "status",
    "sample_stratum",
    "sample_weight",
]


@dataclass(frozen=True)
class PassSpec:
    """Per-pass configuration. Everything else is shared logic."""

    num: int
    outdir: str
    prefix: str
    required_keys: frozenset[str]
    column_order: list[str]


PASSES: dict[int, PassSpec] = {
    1: PassSpec(
        num=1,
        outdir="filipino-emotions-coding",
        prefix="filipino_emotions",
        required_keys=REQUIRED_KEYS,
        column_order=COLUMN_ORDER,
    ),
}

# The filipino-emotions prompt is self-contained (wrapper + Sta. Maria 2010
# framework in one file); only the per-phase case context is inserted.
PROMPT_FILE = "filipino-emotions-prompt-v1.2.txt"
CASE_CONTEXT_SLOT = ("[3–5 neutral, factual sentences on the incident and the publicly known\n"
                     "responses. No judgments about fault.]")
_PROMPT_VERSION_RE = re.compile(r"filipino-emotions-prompt\s+(v[0-9][0-9.]*)", re.I)


def read_prompt() -> tuple[str, str]:
    """Return (prompt text without its '#' metadata header, prompt version)."""
    lines = (PASS_PROMPTS_DIR / PROMPT_FILE).read_text(encoding="utf-8").splitlines()
    header: list[str] = []
    while lines and lines[0].startswith("#"):
        header.append(lines.pop(0))
    match = _PROMPT_VERSION_RE.search("\n".join(header))
    return "\n".join(lines).strip(), (match.group(1) if match else "v1.0")


def build_codebook_prompt(crisis: str, phase: str) -> str:
    """Assemble the prompt: self-contained prompt text + per-phase context."""
    prompt, _version = read_prompt()
    contexts = json.loads((PROMPTS_DIR / "phase-context.json").read_text(encoding="utf-8"))
    case_context = contexts.get(crisis, {}).get(phase, "")
    if not case_context:
        raise RuntimeError(f"No phase context for {crisis}/{phase} in phase-context.json.")

    # The prompt file ends with a "COMMENTS TO CODE / [id | comment text]" stub; the runner
    # appends its own COMMENTS TO CODE section, so drop the stub.
    cut = prompt.rfind("COMMENTS TO CODE")
    if cut != -1:
        prompt = prompt[:cut].rstrip()

    prompt = prompt.replace("MODE: [PILOT | FULL]", "MODE: FULL")
    prompt = prompt.replace(CASE_CONTEXT_SLOT, case_context)

    # Guard: never send a prompt that still contains unfilled template slots.
    if CASE_CONTEXT_SLOT in prompt:
        raise RuntimeError("Prompt still contains the case-context placeholder.")
    leftover = _LEFTOVER_SLOT.search(prompt)
    if leftover:
        raise RuntimeError(
            f"Prompt still contains an unfilled slot {leftover.group(0)!r}."
        )
    return prompt


def to_line(cid: Any, txt: Any) -> str:
    """Format one comment as an `id | text` line (newlines flattened)."""
    flat = str(txt).replace("\r", " ").replace("\n", " ").strip()
    return f"{cid} | {flat}"


# One flag per machine. Explicit flags always override the profile's values.
GPU_PROFILES: dict[str, dict[str, Any]] = {
    # 12 GB desktop card: 4-bit weights (secondary precision), shorter context.
    "rtx3060": {"load_in_4bit": True, "max_model_len": 7168, "concurrency": 2,
                "gpu_memory_utilization": 0.90},
    # 24 GB server card, full-precision (bf16) weights: memory-tight, ~4 at once.
    "l4-full": {"low_memory": True, "max_model_len": 8192, "concurrency": 4,
                "gpu_memory_utilization": 0.95},
    # 24 GB server card, FP8 weights + FP8 KV cache: about half the weight memory, so
    # each step reads half as much and far more comments fit at once. 8-bit, not full
    # precision: disclose it and validate it on the pilot.
    "l4-fp8": {"fp8_weights": True, "fp8_kv_cache": True, "max_model_len": 8192,
               "concurrency": 12, "gpu_memory_utilization": 0.95, "compact_json": True},
    # 40/80 GB: full precision with room to batch.
    "a100": {"max_model_len": 8192, "concurrency": 8, "gpu_memory_utilization": 0.95},
}


def apply_gpu_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Fill unset options from --gpu-profile, then from the global defaults."""
    prof = GPU_PROFILES.get(args.gpu_profile or "", {})
    for key, val in prof.items():
        if getattr(args, key) in (None, False):
            setattr(args, key, val)
    for key, default in (("max_model_len", MAX_MODEL_LEN), ("concurrency", 3)):
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.load_in_4bit and args.fp8_weights:
        raise SystemExit("--load-in-4bit and --fp8-weights are mutually exclusive.")
    return args


def precision_label(args: argparse.Namespace) -> str:
    """Numerical precision of a run, stored with every coded row (do not pool across it)."""
    if args.load_in_4bit:
        w = "nf4"
    elif args.fp8_weights:
        w = "fp8"
    else:
        w = "bf16"
    return f"{w}-weights" + ("+fp8-kv" if args.fp8_kv_cache else "")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI flags. Phases and passes are required (opt-in, no implicit all)."""
    parser = argparse.ArgumentParser(
        description="Code full-phase YouTube comments with a local instruction model."
    )
    parser.add_argument("--crisis", default="ateneo-drowning", help="Crisis slug.")
    parser.add_argument(
        "--phases",
        required=True,
        help="Comma-separated phase slugs, e.g. '04-reckoning' or '01-onset,04-reckoning'.",
    )
    parser.add_argument(
        "--passes",
        required=True,
        help="Comma-separated pass numbers from {1}, e.g. '1'.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows per phase for smoke tests. Default: all rows.",
    )
    parser.add_argument("--chunk-size", type=int, default=5, help="Rows per LLM call.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Generation ceiling per chunk. One coded object is ~400-600 tokens, so "
             "scale this with --chunk-size (e.g. 1024 for size 1, 3072 for size 5).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="nf4 4-bit quantization (bitsandbytes) for <=12GB GPUs. Needed for 9B on 12GB.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=512,
        help="Minimum output tokens guaranteed per comment (default 512). Each request gets "
             "what the context has left, up to --max-new-tokens; comments that leave less "
             "than this are prompt_too_long.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Comments generated at once (default 3). They share the cached codebook prompt; "
             "throughput rises almost linearly while GPU memory allows. 3060 (4-bit): 2; "
             "L4 (full precision): 3-4; A100: 8+.",
    )
    parser.add_argument(
        "--sample-per-phase",
        type=int,
        default=None,
        help="Code a seeded random sample of this many comments per phase (default: all).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20261005,
        help="Seed for --sample-per-phase (same seed = same comments on every machine).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Write to filipino-emotions-coding-<run-name>/ instead (e.g. 'pilot').",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Trim vLLM's unused reservations (implied by --load-in-4bit). Use for full "
             "precision on 24 GB cards such as the L4.",
    )
    parser.add_argument(
        "--gpu-profile",
        choices=sorted(GPU_PROFILES),
        default=None,
        help="Preset for the machine: rtx3060 (4-bit), l4-full (bf16), l4-fp8 (FP8 weights + "
             "FP8 cache), a100 (bf16). Explicit flags override it.",
    )
    parser.add_argument(
        "--fp8-weights",
        action="store_true",
        help="Quantize weights to FP8 at load time (needs an Ada/Hopper GPU such as the L4 "
             "for full speed). 8-bit: disclose and validate on the pilot.",
    )
    parser.add_argument(
        "--fp8-kv-cache",
        action="store_true",
        help="Store the attention cache in FP8: about twice as many comments fit at once. "
             "Changes numerics slightly; disclose.",
    )
    parser.add_argument(
        "--no-language-rule",
        action="store_true",
        help="Keep the model's LANGUAGE as is (by default TAGLISH is corrected to ENGLISH / "
             "FILIPINO when the words clearly show one language; logged in autofix).",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Forbid optional whitespace inside the JSON answer (fewer output tokens, same codes).",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Set VLLM_ATTENTION_BACKEND (e.g. FLASHINFER, TRITON_ATTN) if the default backend "
             "rejects the FP8 cache.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU memory vLLM may use (default 0.90). 0.95 is fine on a "
             "headless server GPU.",
    )
    parser.add_argument(
        "--only-ids",
        type=Path,
        default=None,
        help="File of comment_ids (one per line, or a CSV with a comment_id column) to code; "
             "other comments are skipped. Used for the pilot sample (with --run-name pilot).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Total context (prompt + output). 8192 on large GPUs; 7168 fits a 12GB card "
             "with --load-in-4bit.",
    )
    parser.add_argument(
        "--closing",
        action="store_true",
        help="Append a short 'JSON only' reminder after the comments (off by default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard partial checkpoints and start fresh.",
    )
    return apply_gpu_profile(parser.parse_args(argv))


def split_csv_arg(value: str) -> list[str]:
    """Split a comma-separated flag into cleaned tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def select_sample(df: pd.DataFrame, phase: str, n: Optional[int], seed: int,
                  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Seeded per-phase simple random sample, identical on every machine.

    Weights (`sample_weight` = inverse selection probability) keep phase-level
    estimates unbiased; validate-filipino-emotions.py weights its percentages
    accordingly.
    """
    N = len(df)
    df = df.copy()
    if n is None or n >= N:
        df["sample_stratum"], df["sample_weight"] = "all", 1.0
        return df, {"phase": phase, "N": N, "n": N, "design": "all comments"}
    phase_seed = (seed + zlib.crc32(phase.encode())) % (2**32)
    rnd = df.sample(n=n, random_state=phase_seed)
    w = round(N / n, 4)
    out = rnd.assign(sample_stratum="random", sample_weight=w).sort_values("row_id")
    info = {"phase": phase, "N": N, "n": int(len(out)), "seed": seed, "phase_seed": int(phase_seed),
            "design": "simple random sample",
            "random_n": int(n), "random_weight": w}
    return out, info


def load_phase_df(crisis: str, phase: str, limit: Optional[int]) -> pd.DataFrame:
    """Load a full phase CSV in file order with row_ids for lossless merging."""
    input_path = (
        STAGE_ROOT / ".." / "01-preprocessing" / "output" / crisis / "youtube" / f"{phase}.csv"
    ).resolve()
    full_df = pd.read_csv(input_path)
    if limit is not None:
        full_df = full_df.head(limit)
    df = full_df[["comment_id", "video_id", "text"]].copy()
    df = df.dropna()
    for col in ("comment_id", "video_id", "text"):
        df[col] = df[col].astype(str).str.strip()
    df = df[df["text"] != ""]
    n_before = len(df)
    df = df.drop_duplicates("comment_id", keep="first")  # one coded row per comment
    if len(df) != n_before:
        print(f"  Dropped {n_before - len(df)} duplicate comment_id rows in {phase}.")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


_DEC = json.JSONDecoder()


def load_objs(raw: str) -> list[dict[str, Any]]:
    """Extract every top-level JSON object from the output, ignoring other text."""
    objs: list[dict[str, Any]] = []
    i = 0
    while (i := raw.find("{", i)) != -1:
        try:
            obj, i = _DEC.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    if not objs:
        raise ValueError("no JSON object found")
    return objs


def _enum(v: Any) -> str:
    return str(v).strip().upper().replace(" ", "_")


def _as_bin(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str) and v.strip() in ("0", "1"):
        return int(v.strip())
    raise ValueError(f"not 0/1: {v!r}")


def is_emoji_only(text: str) -> bool:
    """True if the comment has no words once emoji and :shortcodes: are removed."""
    if not text.strip():
        return False
    bare = _UNI_EMOJI.sub("", _SHORTCODE.sub("", text))
    return re.sub(r"[\W_]+", "", bare) == ""


# --- Evidence (filipino-emotions-prompt v1.0) -----------------------------------------
# The prompt asks for free-form "short quote(s) supporting each coded component" and an
# English translation, with no fixed delimiter. Quotes are therefore extracted leniently
# («...» if present, otherwise split on ; | newline ...) and each must appear in the comment.
def _as_tern(v: Any) -> Any:
    """0 / 1, or the string "NA" (screened-out rows)."""
    if isinstance(v, str) and v.strip().upper() == "NA":
        return "NA"
    return _as_bin(v)


_HANDLE_PLACEHOLDER = "[USER]"


class SoftIssue(ValueError):
    """A rejection worth a correction retry, but not worth losing the row over.

    Strict parsing raises it so the row is retried with a correction note. When
    the retries are spent, the row is re-parsed leniently: it is kept, repaired
    where the codebook says how, and given a review flag. Dropping these rows made
    long comments disproportionately missing.
    """

    def __init__(self, msg: str, flag: str) -> None:
        super().__init__(msg)
        self.flag = flag


def _soft(review: list[str], lenient: bool, flag: str, msg: str) -> None:
    if lenient:
        review.append(flag)
    else:
        raise SoftIssue(msg, flag)


_QUOTE_MARKS = re.compile("[\"'`\\‘’“”]")
FUZZY_MIN_RATIO = 0.85  # similarity to the best-matching stretch of the comment


def _norm_text(t: str) -> str:
    """Case-, quote-mark- and whitespace-insensitive form for verbatim matching.

    Quote marks are dropped entirely: comments quoting someone ("...") made the model
    alter or escape them.
    """
    t = _QUOTE_MARKS.sub("", t.replace("️", ""))
    return re.sub(r"\s+", " ", t.casefold()).strip()


def _coverage(quote: str, comment: str) -> float:
    """Similarity of the quote to the stretch of the comment it best lines up with.

    Anchors on the longest shared run of characters, then compares the quote with
    the comment window at that position. Typos and spacing changes score high;
    a quote that is not in the comment has no long shared run and scores low.
    """
    if not quote:
        return 1.0
    sm = SequenceMatcher(None, comment, quote, autojunk=False)
    a, b, size = sm.find_longest_match(0, len(comment), 0, len(quote))
    if size < 4:
        return 0.0
    start = max(0, a - b)
    window = comment[start:start + len(quote) + 3]
    return SequenceMatcher(None, window, quote, autojunk=False).ratio()


_EVID_GUILLEMETS = re.compile(r"«([^«»]+)»")
_EVID_SPLIT = re.compile(r"\s*(?:;|\||\n|…|\.\.\.)\s*")
_EVID_LABEL = re.compile(r"^[A-Z_]{4,}\s*[:=]\s*")  # "FOCAL_EVENT: ..." prefixes
_EVID_ARROW = re.compile(r"\s*(?:->|→|=>).*$")  # "quote -> CODE" suffixes


def evidence_quotes(evidence: str) -> list[str]:
    """Split the model's evidence string into individual quotes."""
    marked = _EVID_GUILLEMETS.findall(evidence)
    if marked:
        return [q.strip() for q in marked if q.strip()]
    out: list[str] = []
    for part in _EVID_SPLIT.split(evidence):
        part = _EVID_ARROW.sub("", _EVID_LABEL.sub("", part.strip())).strip()
        if part:
            out.append(part)
    return out


def _as_evidence_dict(v: Any) -> dict[str, str]:
    """Coerce the model's evidence object to {COMPONENT: quote} (v1.1 object form)."""
    if isinstance(v, dict):
        return {str(k): str(x) for k, x in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "{}"):
            return {}
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return {str(k): str(x) for k, x in parsed.items()}
    raise ValueError(f"bad_type: evidence is {type(v).__name__}, expected object keyed by {list(EVIDENCE_KEYS)}")


def check_evidence(
    obj: dict[str, Any], text: str, lenient: bool = False, review: Optional[list[str]] = None
) -> None:
    """Screened-in comments need per-component quotes; every quote must be words from the comment (v1.1)."""
    review = review if review is not None else []
    try:
        evidence = _as_evidence_dict(obj.get("evidence"))
        evidence_en = _as_evidence_dict(obj.get("evidence_en"))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    obj["evidence"] = evidence
    obj["evidence_en"] = evidence_en
    if obj["NEG_SOCIAL_EMOTION"] != 1:
        return
    if not evidence:
        _soft(review, lenient, "evidence_missing",
              "evidence_missing: NEG_SOCIAL_EMOTION is 1, so give per-component quotes from the comment")
        return
    # Required keys per the prompt: FOCAL_EVENT unless UNCLEAR, plus each coded
    # non-default component and SARCASM when 1.
    required: set[str] = set()
    if obj.get("FOCAL_EVENT") not in ("UNCLEAR", "NA"):
        required.add("FOCAL_EVENT")
    if obj.get("BLAME_LOCUS") in ("OTHER", "SELF"):
        required.add("BLAME_LOCUS")
    if obj.get("CONTROL_STRATEGY") in ("IGNORE", "REFRAME"):
        required.add("CONTROL_STRATEGY")
    if obj.get("ACTION_TENDENCY") not in ("NONE_EXPRESSED", "NA"):
        required.add("ACTION_TENDENCY")
    if obj.get("REGULATION") in ("INTENSE_DISTRESS", "NONVERBAL_ONLY"):
        required.add("REGULATION")
    if obj.get("RELATIONSHIP_LEVEL") not in ("NOT_DETERMINABLE", "NA"):
        required.add("RELATIONSHIP_LEVEL")
    if obj.get("SARCASM") == 1:
        required.add("SARCASM")
    missing = required - set(evidence)
    if missing:
        _soft(review, lenient, "evidence_missing",
              f"evidence_missing: no quote for {sorted(missing)}")
    comment = _norm_text(_HANDLE.sub(_HANDLE_PLACEHOLDER, text))
    for comp, quote in evidence.items():
        if comp not in EVIDENCE_KEYS:
            raise ValueError(f"bad_evidence_key: {comp!r} not in {list(EVIDENCE_KEYS)}")
        q = _norm_text(str(quote)).strip(" .,!?;:")
        if not q or q in comment:
            continue
        if _coverage(q, comment) >= FUZZY_MIN_RATIO:
            review.append("evidence_fuzzy")  # typo / spacing differences only
        else:
            _soft(review, lenient, "evidence_unverified",
                  f"evidence_not_verbatim: {comp} «{str(quote)[:60]}» is not in the comment")
    if set(evidence_en) != set(evidence):
        _soft(review, lenient, "evidence_en_missing",
              f"evidence_en_mismatch: evidence keys {sorted(evidence)} vs evidence_en keys {sorted(evidence_en)}; give the same keys")


# Common Filipino and Cebuano function words. Used only to FLAG likely LANGUAGE
# errors for review, never to overwrite the model's code.
_PH_MARKERS = frozenset(
    """ang ng nang mga sa si ni na po opo ko mo niya nila namin natin kami kayo sila siya
    ito iyan yan iyon yun yung naman lang lamang talaga kasi pero hindi di wala may meron
    ba pa din rin daw raw sana dapat kaya ano bakit sino grabe sobra nga ug dili gyud kaayo
    unsa kini kana walay""".split()
)
_WORD = re.compile(r"[a-zA-Z']+")


# --- Deterministic codebook rules applied after coding (no prompt change) ---------
# Toggle set by main(); --no-language-rule turns the A1 rule off.
LANGUAGE_RULE = True

_EN_MARKERS = frozenset(
    """the is are was were and to of you your for that this with not but he she they we
    it in on be have has should why like will so them their what who how can just all
    because been being do does did there these those an if or my our his her its would
    could must""".split()
)
_REPLY_START = re.compile(r"^\s*(?:@|\[user\]|user_)", re.I)


def is_reply(comment_id: str, text: str) -> bool:
    """YouTube reply ids are '<parent>.<reply>'; replies also tend to open with a mention."""
    return "." in str(comment_id) or bool(_REPLY_START.match(str(text)))


def language_rule(language: str, text: str) -> Optional[str]:
    """Correct TAGLISH, the value the model over-uses, when the words clearly show one
    language: no Filipino words and 2+ English function words -> EN; no English and 2+
    Filipino -> FIL. Anything ambiguous keeps the model's code."""
    if language != "TAGLISH":
        return None
    words = [w.lower().strip("'") for w in _WORD.findall(_SHORTCODE.sub(" ", text))]
    fil = sum(w in _PH_MARKERS for w in words)
    en = sum(w in _EN_MARKERS for w in words)
    if fil == 0 and en >= 2:
        return "EN"
    if en == 0 and fil >= 2:
        return "FIL"
    return None


def language_flag(language: str, text: str) -> Optional[str]:
    """Return an autofix flag when LANGUAGE disagrees with a simple word check."""
    words = {w.lower().strip("'") for w in _WORD.findall(_SHORTCODE.sub(" ", text))}
    if not words:
        return None
    has_ph = bool(words & _PH_MARKERS)
    if language in ("TAGLISH", "FIL") and not has_ph:
        return "language_check:no_filipino_words"
    if language == "EN" and has_ph:
        return "language_check:filipino_words_in_english"
    return None


# --- EMOTION_WORDS: literal terms, in any inflected or affixed form ----------------
# Advisory only (review flags). Terms that are short or common as substrings (asar, inis,
# suya) must start at a word boundary, optionally after a verbal prefix.
_PREFIX = r"(?:na|nag|mag|naka|nakaka|nai|mai|ka|pa)?-?"
_EMOTION_RE = {
    "ASAR": re.compile(rf"\b{_PREFIX}asar", re.I),
    "BUWISIT": re.compile(r"buwisit", re.I),
    "GALIT": re.compile(r"galit", re.I),
    "INGGIT": re.compile(r"inggit", re.I),
    "INIS": re.compile(rf"\b{_PREFIX}inis", re.I),
    "MUHI": re.compile(r"muhi", re.I),
    "NGITNGIT": re.compile(r"ngitngit", re.I),
    "PIKON": re.compile(r"pikon", re.I),
    "POOT": re.compile(r"poot", re.I),
    "SAMA_NG_LOOB": re.compile(r"sama\s+ng\s+loob|masama\s+ang\s+loob|masamang\s+loob", re.I),
    "SELOS": re.compile(r"selos", re.I),
    "SUKLAM": re.compile(r"suklam", re.I),
    "SUYA": re.compile(rf"\b{_PREFIX}suya", re.I),
    "TAMPO": re.compile(r"tampo", re.I),
    "YAMOT": re.compile(r"yamot", re.I),
}


def emotion_terms_in(text: str) -> list[str]:
    return [t for t, pat in _EMOTION_RE.items() if pat.search(text)]


# --- Cluster logic (Step 3 and CLUSTER_CONFLICT) ----------------------------------
# FOCAL_EVENT sets the default cluster.
DEFAULT_CLUSTER = {"MORAL_VIOLATION": "C1", "UNMET_EXPECTATION": "C2",
                   "HINDRANCE": "C3", "UNCLEAR": "UNCLASSIFIED"}
# Which cluster(s) each coded component value describes in the FRAMEWORK section.
# BLAME SELF fits C2 and C3 (both: "the actor is to blame"); IGNORE fits C1 and C3;
# CONTROLLED fits C2 and C3. Values absent here (NONE_EXPRESSED, NOT_EXPRESSED,
# NOT_DETERMINABLE) are not coded and never conflict.
COMPONENT_CLUSTERS: dict[str, dict[str, frozenset[str]]] = {
    "BLAME_LOCUS": {"OTHER": frozenset({"C1"}), "SELF": frozenset({"C2", "C3"})},
    "CONTROL_STRATEGY": {"IGNORE": frozenset({"C1", "C3"}), "REFRAME": frozenset({"C2"})},
    "ACTION_TENDENCY": {"LASTING_WITHDRAWAL": frozenset({"C1"}),
                        "RELATIONSHIP_MAINTAINED": frozenset({"C2"}),
                        "TEMPORARY_DISTANCE": frozenset({"C3"})},
    "REGULATION": {"INTENSE_DISTRESS": frozenset({"C1"}), "CONTROLLED": frozenset({"C2", "C3"}),
                   "NONVERBAL_ONLY": frozenset({"C3"})},
    "RELATIONSHIP_LEVEL": {"KAPWA": frozenset({"C1"}), "HINDI_IBANG_TAO": frozenset({"C2"}),
                           "IBANG_TAO": frozenset({"C3"})},
}


def component_conflict(obj: dict[str, Any]) -> Optional[int]:
    """CLUSTER_CONFLICT as the prompt defines it: 1 if any coded component matches a
    different cluster than CLUSTER. None when CLUSTER is not C1-C3 (nothing to compare)."""
    if obj["CLUSTER"] not in ("C1", "C2", "C3"):
        return None
    for key, table in COMPONENT_CLUSTERS.items():
        clusters = table.get(obj[key])
        if clusters is not None and obj["CLUSTER"] not in clusters:
            return 1
    return 0


def validate_record(
    obj: dict[str, Any], text: str, lenient: bool = False, review: Optional[list[str]] = None
) -> list[str]:
    """Check one coded record against filipino-emotions-prompt v1.1 and normalise it in place.

    Returns deterministic autofixes. Review flags (rows kept but needing a look)
    are appended to `review`; see SoftIssue.
    """
    fixes: list[str] = []
    review = review if review is not None else []

    # Free-text fields: strings only, and never the template or our instructions.
    for k in FREE_TEXT_KEYS:
        v = obj.get(k, "")
        if v is None:
            v = ""
        if isinstance(v, (list, dict)):
            raise ValueError(f"bad_type: {k} is {type(v).__name__}")
        v = str(v)
        low = v.lower()
        if any(m in low for m in PLACEHOLDER_MARKERS):
            raise ValueError(f"placeholder_leak: {k}={v[:70]!r}")
        obj[k] = v
    # v1.1 object-form evidence: keys are components, values are exact quotes.
    for k in EVIDENCE_DICT_KEYS:
        try:
            ev = _as_evidence_dict(obj.get(k))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        for comp, quote in ev.items():
            if comp not in EVIDENCE_KEYS:
                raise ValueError(f"bad_evidence_key: {k} has {comp!r}")
            low = str(quote).lower()
            if any(m in low for m in PLACEHOLDER_MARKERS):
                raise ValueError(f"placeholder_leak: {k}.{comp}={str(quote)[:70]!r}")
        obj[k] = ev
    for k in EVIDENCE_DICT_KEYS:  # general rule 6
        fixed: dict[str, str] = {}
        changed = False
        for comp, quote in obj[k].items():
            q = str(quote)
            if _HANDLE.search(q):
                q = _HANDLE.sub(_HANDLE_PLACEHOLDER, q)
                changed = True
            fixed[comp] = q
        if changed:
            obj[k] = fixed
            fixes.append("anonymized_handle")

    for k in BINARY_KEYS:
        try:
            obj[k] = _as_bin(obj[k])
        except ValueError as exc:
            raise ValueError(f"bad_binary: {k} {exc}") from exc
    for k in TERNARY_KEYS:
        try:
            obj[k] = _as_tern(obj[k])
        except ValueError as exc:
            raise ValueError(f"bad_ternary: {k} {exc}") from exc

    for k, allowed in SCALAR_ENUMS.items():
        val = _enum(obj[k])
        if val not in allowed:
            raise ValueError(f"bad_enum: {k}={obj[k]!r}")
        obj[k] = val

    v = obj["EMOTION_WORDS"]
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not v:
        raise ValueError(f"bad_list: EMOTION_WORDS={obj['EMOTION_WORDS']!r}")
    words: list[str] = []
    for x in v:
        e = _enum(x)
        if e not in EMOTION_WORD_VALUES:
            raise ValueError(f"bad_enum: EMOTION_WORDS has {x!r}")
        if e not in words:
            words.append(e)

    # Step 1: a screened-out comment is NA everywhere; a screened-in one has no NA.
    coded = (*COMPONENT_KEYS, *TERNARY_KEYS)
    if obj["NEG_SOCIAL_EMOTION"] == 0:
        if words != ["NA"] or any(obj[k] != "NA" for k in coded):
            fixes.append("screened_out->NA")
        for k in coded:
            obj[k] = "NA"
        obj["EMOTION_WORDS"] = ["NA"]
        for k in EVIDENCE_DICT_KEYS:
            if obj.get(k) != {}:
                obj[k] = {}
                if "screened_out_evidence->{}" not in fixes:
                    fixes.append("screened_out_evidence->{}")
    else:
        words = [w for w in words if w != "NA"]
        if len(words) > 1 and "NONE" in words:
            words.remove("NONE")  # NONE alongside real terms is contradictory
        na_fields = [k for k in coded if obj[k] == "NA"] + ([] if words else ["EMOTION_WORDS"])
        if na_fields:
            raise ValueError(
                f"bad_na: NEG_SOCIAL_EMOTION is 1 but {na_fields} are NA; give each a codebook "
                "value (EMOTION_WORDS: the literal terms, or NONE)"
            )
        obj["EMOTION_WORDS"] = words

    # Evidence must be words that really appear in the comment.
    check_evidence(obj, text, lenient, review)

    # General rule 1: NEEDS_CONTEXT is only for replies. A top-level comment cannot
    # depend on a parent comment.
    if obj["NEEDS_CONTEXT"] == 1 and not is_reply(obj.get("id", ""), text):
        obj["NEEDS_CONTEXT"] = 0
        fixes.append("needs_context->0(not_reply)")

    # LANGUAGE: correct TAGLISH when the words clearly show one language
    # (--no-language-rule: off); otherwise only flag disagreements.
    if LANGUAGE_RULE and not is_emoji_only(text):
        new_lang = language_rule(obj["LANGUAGE"], text)
        if new_lang:
            fixes.append(f"language->{new_lang}(rule)")
            obj["LANGUAGE"] = new_lang
    flag = language_flag(obj["LANGUAGE"], text)
    if flag and not is_emoji_only(text):
        fixes.append(flag)  # flagged for review, not overwritten

    if obj["NEG_SOCIAL_EMOTION"] == 1:
        # EMOTION_WORDS is literal: flag terms that are not in the text, and terms in the
        # text that were not listed. Advisory; the model's list is kept.
        listed = {w for w in obj["EMOTION_WORDS"] if w != "NONE"}
        found = set(emotion_terms_in(text))
        for t in sorted(listed - found):
            review.append(f"emotion_word_not_in_text({t})")
        for t in sorted(found - listed):
            review.append(f"emotion_word_missed({t})")

        # Step 3: CLUSTER follows FOCAL_EVENT unless the model overrode it (allowed only
        # when the focal event is ambiguous, and explained in NOTE).
        default = DEFAULT_CLUSTER[obj["FOCAL_EVENT"]]
        if obj["CLUSTER"] != default:
            review.append(f"cluster_override({default}->{obj['CLUSTER']})")
            if not obj["NOTE"].strip():
                review.append("cluster_override_no_note")
        # CLUSTER_CONFLICT is defined by the coded components; flag disagreements with the
        # model's value (kept as coded, not overwritten).
        expected = component_conflict(obj)
        if expected is not None and obj["CLUSTER_CONFLICT"] != expected:
            review.append(f"cluster_conflict_check(model={obj['CLUSTER_CONFLICT']},rule={expected})")
    return fixes


def parse_chunk_response(
    raw: str,
    row_ids: list[int],
    required: frozenset[str],
    id_of: Optional[dict[int, str]] = None,
    text_of: Optional[dict[int, str]] = None,
    lenient: bool = False,
) -> list[dict[str, Any]]:
    """Parse one chunk response into per-row records (lenient: keep soft issues as flags)."""
    # Strip hallucinated instruction loops before parsing
    loop_marker = "Now output exactly one JSON object per comment above"
    if loop_marker in raw:
        raw = raw.split(loop_marker)[0]

    if not raw.strip():
        raise ValueError("empty_response")
    try:
        objs = load_objs(raw)
    except Exception as exc:
        raise ValueError(f"json_decode: {exc!r}") from exc
    if len(objs) > len(row_ids):
        objs = objs[: len(row_ids)]  # model kept going; the id check guards correctness
    if len(objs) != len(row_ids):
        raise ValueError(f"count_mismatch: {len(objs)} objs for {len(row_ids)} rows")
    records: list[dict[str, Any]] = []
    for rid, obj in zip(row_ids, objs):
        review: list[str] = []
        missing = required - set(obj.keys())
        if missing:
            raise ValueError(f"missing_keys: {sorted(missing)}")
        if id_of is not None:
            want = str(id_of.get(rid, "")).strip()
            got = str(obj.get("id", "")).strip()
            if got != want:
                raise ValueError(f"id_mismatch: {got!r} vs {want!r}")
        try:
            fixes = validate_record(obj, (text_of or {}).get(rid, ""), lenient, review)
        except SoftIssue as exc:
            raise SoftIssue(f"row {rid}: {exc}", exc.flag) from exc
        except ValueError as exc:
            raise ValueError(f"row {rid}: {exc}") from exc
        obj["autofix"] = ";".join(fixes)
        obj["review_flags"] = ";".join(dict.fromkeys(review))
        obj["_row_id"] = rid
        records.append(obj)
    return records


class LocalGenerator:
    """Thin wrapper around vLLM for generation with prefix caching (codebook prompt reused)."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        load_in_4bit: bool = False,
        max_model_len: int = MAX_MODEL_LEN,
        low_memory: bool = False,
        gpu_memory_utilization: Optional[float] = None,
        concurrency: int = 1,
        min_new_tokens: int = 512,
        fp8_weights: bool = False,
        fp8_kv_cache: bool = False,
        compact_json: bool = False,
        attention_backend: Optional[str] = None,
    ) -> None:
        import os

        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend  # read at vLLM import
        from vllm import LLM, SamplingParams

        self.compact_json = compact_json

        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min(min_new_tokens, max_new_tokens)
        self.max_model_len = max_model_len

        print(f"\nInitializing vLLM Engine for '{model_id}'...")
        print(" -> Prefix Caching: ENABLED (codebook prompt is computed once and reused)")
        weights = "nf4 (bitsandbytes 4-bit)" if load_in_4bit else ("FP8" if fp8_weights else "bf16 (full)")
        print(f" -> Weights: {weights} | KV cache: {'FP8' if fp8_kv_cache else 'auto (bf16)'}"
              f" | compact JSON: {'ON' if compact_json else 'OFF'}")

        engine_kwargs: dict[str, Any] = {"gpu_memory_utilization": 0.90,
                                         "max_num_seqs": max(1, concurrency)}
        if load_in_4bit:
            engine_kwargs["quantization"] = "bitsandbytes"
        elif fp8_weights:
            engine_kwargs["quantization"] = "fp8"  # online dynamic FP8 quantization
        if fp8_kv_cache:
            engine_kwargs["kv_cache_dtype"] = "fp8"
        if load_in_4bit or low_memory:
            # Low-memory profile (12 GB cards in 4-bit; 24 GB cards such as the L4 at
            # full precision). The script sends one request at a time, so vLLM's
            # defaults (256 parallel sequences, 8192-token prefill batches) only reserve
            # memory that is never used. Shrinking them frees room for the 8K KV cache
            # without changing what the model computes.
            engine_kwargs.update(
                max_num_seqs=max(1, concurrency),  # only as many sequences as we send
                max_num_batched_tokens=512,  # small prefill pieces = small activation peak
                enforce_eager=True,  # skip CUDA graph capture (saves memory, some speed)
            )
        if gpu_memory_utilization is not None:
            # 0.90 on a card that also drives a desktop (3060); up to 0.95 on a headless
            # server GPU (L4, A100) where nothing else uses the memory.
            engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        print(f" -> Memory: utilization {engine_kwargs['gpu_memory_utilization']}, "
              f"low-memory profile {'ON' if (load_in_4bit or low_memory) else 'OFF'}, "
              f"concurrency {concurrency}")

        self.llm = LLM(
            model=model_id,
            enable_prefix_caching=True,
            max_model_len=self.max_model_len,
            **engine_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._SamplingParams = SamplingParams
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )

    def _chat(self, prompt: str) -> str:
        """Wrap the prompt in the model's chat template (raw prompt if it has none)."""
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            print("WARNING: chat template failed; sending the raw prompt. "
                  "Is this a base (non-instruct) model?")
            return prompt

    def count_prompt_tokens(self, prompt: str) -> int:
        """Tokens the prompt occupies once wrapped in the chat template."""
        return len(self.tokenizer(self._chat(prompt), add_special_tokens=False)["input_ids"])

    def _schema_params(self, schema: dict[str, Any], max_tokens: Optional[int] = None) -> Any:
        """Greedy sampling constrained to `schema` (vLLM >= 0.11 name, older fallback)."""
        max_tokens = max_tokens or self.max_new_tokens
        try:
            from vllm.sampling_params import StructuredOutputsParams

            try:
                so = StructuredOutputsParams(json=schema, disable_any_whitespace=self.compact_json)
            except TypeError:  # older field set
                so = StructuredOutputsParams(json=schema)
            return self._SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                structured_outputs=so,
            )
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            return self._SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                guided_decoding=GuidedDecodingParams(json=schema),
            )

    def output_budget(self, prompt: str, prefill: str = "") -> int:
        """Output tokens available for this prompt: what the context has left, capped
        at --max-new-tokens. Raises prompt_too_long below --min-new-tokens.

        A fixed 1,024-token reservation left only ~140 tokens for the comment on a
        7,168 context once the v1.5 prompt reached ~6,000 tokens. Most answers need
        far less, so each request gets the room that is actually left.
        """
        room = self.max_model_len - self.count_prompt_tokens(prompt) - len(prefill) // 2
        if room < self.min_new_tokens:
            # No digits in the message: the retry logic treats "500" in an error as transient.
            raise ValueError("prompt_too_long: comment does not fit in the context window")
        return min(self.max_new_tokens, room)

    def generate_batch(
        self, jobs: list[tuple[str, str, Optional[dict[str, Any]]]]
    ) -> list[Any]:
        """Generate several (prompt, prefill, schema) jobs in one vLLM call.

        Decoding speed is limited by reading the model weights once per step, so
        a step for several sequences costs about the same as for one; the shared
        codebook prompt is cached once. Returns, per job, the text or the
        Exception for that job (e.g. prompt too long). An engine-level failure
        raises for the whole batch.
        """
        results: list[Any] = [None] * len(jobs)
        chats, params, where = [], [], []
        for i, (prompt, prefill, schema) in enumerate(jobs):
            if schema is not None and prefill:
                results[i] = ValueError("prefill cannot be combined with a schema")
                continue
            try:
                budget = self.output_budget(prompt, prefill)
            except ValueError as exc:
                results[i] = exc
                continue
            chats.append(self._chat(prompt) + prefill)
            params.append(self._schema_params(schema, budget) if schema is not None
                          else self._SamplingParams(temperature=0.0, max_tokens=budget))
            where.append(i)
        if chats:
            outputs = self.llm.generate(chats, sampling_params=params, use_tqdm=False)
            for i, out in zip(where, outputs):
                text = jobs[i][1] + out.outputs[0].text
                results[i] = TruncatedOutput(text) if out.outputs[0].finish_reason == "length" else text
        return results

    def generate(
        self, prompt: str, prefill: str = "", schema: Optional[dict[str, Any]] = None
    ) -> str:
        """Deterministic generation.

        With `schema`, decoding is constrained to that JSON schema (all keys present,
        codebook values only) and `prefill` must be empty. Without it, `prefill` is
        forced as the start of the reply.
        """
        if schema is not None and prefill:
            raise ValueError("prefill cannot be combined with a schema")
        chat = self._chat(prompt) + prefill
        budget = self.output_budget(prompt, prefill)
        params = (self._schema_params(schema, budget) if schema is not None
                  else self._SamplingParams(temperature=0.0, max_tokens=budget))
        outputs = self.llm.generate(
            [chat],
            sampling_params=params,
            use_tqdm=False,  # no per-call progress bar
        )
        text = prefill + outputs[0].outputs[0].text
        if outputs[0].outputs[0].finish_reason == "length":
            raise TruncatedOutput(text)
        return text


def run_pair(
    df: pd.DataFrame,
    spec: PassSpec,
    crisis: str,
    phase: str,
    model_id: str,
    chunk_size: int,
    max_new_tokens: int,
    overwrite: bool,
    generator: Optional[LocalGenerator] = None,
    use_closing: bool = False,
    concurrency: int = 1,
    run_name: Optional[str] = None,
    sample_info: Optional[dict[str, Any]] = None,
    precision: str = "bf16-weights",
) -> bool:
    """Code one phase x one pass with checkpoint resume. Returns True if clean."""
    outdir = spec.outdir + (f"-{run_name}" if run_name else "")
    pass_dir = OUTPUT_ROOT / crisis / outdir / phase
    pass_dir.mkdir(parents=True, exist_ok=True)
    output_path = pass_dir / f"{spec.prefix}_results.jsonl"
    log_path = pass_dir / f"{spec.prefix}_failures.log"
    pending_path = pass_dir / f"{spec.prefix}_pending.log"
    csv_path = pass_dir / f"{spec.prefix}_results.csv"
    debug_path = pass_dir / "debug_raw.jsonl"
    partial_resp = Path(str(output_path) + ".partial.jsonl")
    partial_fail = Path(str(log_path) + ".partial.log")
    partial_pending = Path(str(pending_path) + ".partial.log")

    prompt_template = build_codebook_prompt(crisis, phase)
    if sample_info is not None:
        (pass_dir / "sample_info.json").write_text(json.dumps(sample_info, indent=2), encoding="utf-8")
    _, prompt_version = read_prompt()
    (pass_dir / "run_info.json").write_text(json.dumps({
        "model": model_id, "precision": precision, "prompt_version": prompt_version,
        "schema": SCHEMA_VERSION,
        "concurrency": concurrency, "max_model_len": getattr(generator, "max_model_len", None),
        "max_new_tokens": max_new_tokens, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"Filipino-emotions prompt {prompt_version} (self-contained, Sta. Maria 2010)")

    print(f"\n\n{'='*80}\nPROMPT FOR PHASE: {phase}\n{'='*80}\n{prompt_template[:500]}...[TRUNCATED FOR LOGS]\n{'='*80}\n")
    id_of = dict(zip(df["row_id"].tolist(), df["comment_id"].astype(str).tolist()))
    text_of = dict(zip(df["row_id"].tolist(), df["text"].astype(str).tolist()))
    retry_notes: dict[tuple[int, ...], str] = {}  # latest correction note per single row
    correction_counts: dict[tuple[int, ...], int] = {}

    chunks_queue = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    raw_responses: list[dict[str, Any]] = []
    hard_failures: list[dict[str, Any]] = []
    revisit_counts: dict[tuple[int, ...], int] = {}
    outage_streak = 0

    # Resume: successes filter the queue; prior failures are retried, never skipped.
    if not overwrite:
        try:
            ok_ids: set[int] = set()
            stale_version = 0
            frame_rids = set(df["row_id"].tolist())
            if partial_resp.exists():
                with open(partial_resp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        # Outputs made with another prompt version are recoded, not
                        # mixed in (v1.0 checkpoints carry no version field).
                        if rec.get("prompt_version", "v1.0") != prompt_version or rec.get("schema") != SCHEMA_VERSION:
                            stale_version += 1
                            continue
                        # Rows coded at another precision (nf4 / fp8 / bf16) are recoded too.
                        if rec.get("precision", precision) != precision:
                            stale_version += 1
                            continue
                        # Only rows in this run's frame (a sample, or --only-ids).
                        if not set(rec.get("row_ids", [])) <= frame_rids:
                            continue
                        ok_ids.update(rec.get("row_ids", []))
                        raw_responses.append(rec)
            prior_failures = 0
            if partial_fail.exists():
                with open(partial_fail, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if "row_id" in entry:
                                prior_failures += 1
                        except Exception:
                            pass
                # Clear the stale failure checkpoint so retries append cleanly.
                try:
                    partial_fail.unlink()
                except Exception:
                    pass
            pending_prior = 0
            if partial_pending.exists():
                with open(partial_pending, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if "row_id" in entry:
                                pending_prior += 1
                        except Exception:
                            pass
            hard_failures = []
            if stale_version:
                print(
                    f"Ignoring {stale_version} checkpointed rows coded with another prompt "
                    f"version or precision; they will be recoded ({prompt_version}, {precision})."
                )
            if ok_ids:
                chunks_queue = [
                    c
                    for c in chunks_queue
                    if not set(c["row_id"].tolist()) <= ok_ids
                ]
                print(
                    f"Resuming: {len(ok_ids)} rows done, "
                    f"{prior_failures} prior failures to retry, "
                    f"{pending_prior} pending-transient, "
                    f"{len(chunks_queue)} chunks left."
                )
        except Exception as exc:
            print(f"Resume check failed ({exc!r}); starting fresh.")
    else:
        for p in (partial_resp, partial_fail, partial_pending, debug_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Starting dynamic queue with {len(chunks_queue)} initial chunks...")
    owned_generator = False
    if generator is None:
        generator = LocalGenerator(model_id, max_new_tokens)
        owned_generator = True
    assert generator is not None

    # Pre-flight: if the fixed part of the prompt plus the output budget cannot fit
    # in the context window, every row would fail and be logged as a "hard failure".
    # Abort loudly instead of producing an empty run.
    base_tokens = generator.count_prompt_tokens(prompt_template + "\n\nCOMMENTS TO CODE\n")
    min_out = getattr(generator, "min_new_tokens", generator.max_new_tokens)
    budget = generator.max_model_len - min_out
    print(
        f"Prompt template: {base_tokens} tokens | max_model_len {generator.max_model_len} | "
        f"output {min_out}-{generator.max_new_tokens} tokens per comment | up to "
        f"{budget - base_tokens} tokens of comment text "
        f"({generator.max_model_len - generator.max_new_tokens - base_tokens} with the full output budget)."
    )
    if base_tokens >= budget:
        raise RuntimeError(
            f"Prompt template ({base_tokens} tokens) does not fit: only {budget} tokens are "
            f"available after reserving {min_out} for output. Use a larger --max-model-len "
            f"(8192 on the L4/A100) or a lower --min-new-tokens."
        )

    pbar = tqdm(
        total=len(df),
        initial=len(df) - sum(len(c) for c in chunks_queue),
        desc=f"LLM-local {crisis}/{phase}/pass{spec.num}",
        unit="rows",
    )

    def split_or_log(chunk: pd.DataFrame, n_rows: int, exc: Exception) -> None:
        """Non-outage failure: halve the chunk, or log a single-row content failure."""
        if n_rows > 1:
            print(f"Chunk failure (size {n_rows}): {exc!r} — splitting.")
            mid = n_rows // 2
            chunks_queue.insert(0, chunk.iloc[mid:])
            chunks_queue.insert(0, chunk.iloc[:mid])
        else:
            row = chunk.iloc[0]
            print(f"Hard failure on row_id {row['row_id']}: {exc!r}")
            entry = {
                "row_id": int(row["row_id"]),
                "comment_id": row.get("comment_id"),
                "reason": "content_failure",
                "error": repr(exc),
                "text_length": len(str(row["text"])),
            }
            hard_failures.append(entry)
            with open(partial_fail, "a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "api_failure", **entry}, ensure_ascii=False) + "\n")
            pbar.update(1)

    try:
        def build_job(current: pd.DataFrame) -> tuple[str, str, Optional[dict[str, Any]]]:
            row_ids = current["row_id"].tolist()
            payload = "\n".join(to_line(r["comment_id"], r["text"]) for _, r in current.iterrows())
            # One comment per call: constrain decoding to the codebook schema, which
            # guarantees every key and only codebook values. Multi-comment chunks fall
            # back to the old forced-prefix approach.
            schema = build_record_schema(str(current.iloc[0]["comment_id"])) if len(current) == 1 else None
            prefill = "" if schema else '{"id":"' + str(current.iloc[0]["comment_id"]) + '",'
            full_prompt = (
                prompt_template
                + "\n\nCOMMENTS TO CODE\n"
                + payload
                + (CLOSING if use_closing else "")
                + retry_notes.get(tuple(row_ids), "")
            )
            return full_prompt, prefill, schema

        def accept(current: pd.DataFrame, raw: str, lenient: bool = False) -> None:
            nonlocal outage_streak
            rec = {"row_ids": current["row_id"].tolist(), "text": raw or "",
                   "prompt_version": prompt_version, "schema": SCHEMA_VERSION,
                   "precision": precision}
            if lenient:
                rec["lenient"] = True
            raw_responses.append(rec)
            with open(partial_resp, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outage_streak = 0
            pbar.update(len(current))

        def handle_error(current: pd.DataFrame, exc: Exception) -> bool:
            """Outage / generation error for one job. Returns True to stop the phase."""
            nonlocal outage_streak
            n_rows, key = len(current), tuple(current["row_id"].tolist())
            if is_rate_limited(exc) or is_transient(exc):
                reason = "rate_limited_exhausted" if is_rate_limited(exc) else "transient_exhausted"
                outage_streak += 1
                revisit_counts[key] = revisit_counts.get(key, 0) + 1
                if revisit_counts[key] > MAX_REVISITS:
                    print(f"   -> {reason} after {MAX_REVISITS} requeues. Marking {n_rows} rows pending.")
                    append_pending_rows(partial_pending, current, reason, repr(exc))
                    pbar.update(n_rows)
                else:
                    print(f"   -> {reason}. Requeueing chunk ({revisit_counts[key]}/{MAX_REVISITS}); "
                          f"cooldown {COOLDOWN_S}s.")
                    time.sleep(COOLDOWN_S)
                    chunks_queue.append(current)
                if outage_streak >= BREAKER_TRIP:
                    print(f"Circuit breaker: {outage_streak} consecutive outage-exhausted chunks. "
                          f"Aborting with {len(chunks_queue)} chunks queued — partials saved, rerun to resume.")
                    for c in chunks_queue:
                        append_pending_rows(partial_pending, c, "breaker_abort", repr(exc))
                    chunks_queue.clear()
                    return True
                return False
            # Non-outage error (e.g. prompt too long): split or log it.
            split_or_log(current, n_rows, exc)
            return False

        def handle_output(current: pd.DataFrame, raw: str,
                          pre_exc: Optional[Exception] = None) -> None:
            """Validate one answer: accept, retry with a correction note, keep with flags, or log."""
            n_rows, row_ids = len(current), current["row_id"].tolist()
            key = tuple(row_ids)
            try:
                if pre_exc is not None:
                    raise pre_exc
                parse_chunk_response(raw or "", row_ids, spec.required_keys, id_of, text_of)
            except Exception as exc:
                dump_raw(debug_path, row_ids, repr(exc), raw or "")
                if n_rows == 1 and correction_counts.get(key, 0) < MAX_CORRECTIONS:
                    # Greedy decoding would repeat itself, so retry with the rejection
                    # reason appended to change the input.
                    correction_counts[key] = correction_counts.get(key, 0) + 1
                    print(f"Row {row_ids[0]} rejected ({str(exc)[:120]}); retry "
                          f"{correction_counts[key]}/{MAX_CORRECTIONS} with a correction note.")
                    retry_notes[key] = (
                        "\n\nCORRECTION: the previous answer for this comment was rejected: "
                        f"{str(exc)[:200]}. Code only this comment, from its own words. If "
                        "NEG_SOCIAL_EMOTION is 0, set every other coded field to NA (EMOTION_WORDS "
                        "[\"NA\"]) and evidence and evidence_en to {}. If it is 1, no field may be NA: use NONE_EXPRESSED / "
                        "NOT_DETERMINABLE / UNCLASSIFIED when in doubt, and EMOTION_WORDS "
                        "[\"NONE\"] when no listed term appears. Evidence and evidence_en are objects keyed by "
                        "component (FOCAL_EVENT, BLAME_LOCUS, CONTROL_STRATEGY, ACTION_TENDENCY, "
                        "REGULATION, RELATIONSHIP_LEVEL, SARCASM) with exact quotes from the comment and "
                        "their English translations under the same keys."
                        + (" Your answer was too long: keep evidence and evidence_en to a few short "
                           "quotes, and quote a repeated emoji or phrase only once."
                           if isinstance(exc, TruncatedOutput) else "")
                    )
                    chunks_queue.insert(0, current)
                    return
                # Retries spent: keep the row if only soft issues remain (flagged).
                if n_rows == 1:
                    try:
                        parse_chunk_response(raw or "", row_ids, spec.required_keys, id_of,
                                             text_of, lenient=True)
                    except Exception:
                        split_or_log(current, n_rows, exc)
                        return
                    print(f"Row {row_ids[0]} kept with review flags after {MAX_CORRECTIONS} retries.")
                    accept(current, raw, lenient=True)
                    return
                split_or_log(current, n_rows, exc)
                return
            accept(current, raw)

        while chunks_queue:
            batch = [chunks_queue.pop(0) for _ in range(min(concurrency, len(chunks_queue)))]
            results = generate_batch_with_retry(generator, [build_job(c) for c in batch])
            if any(isinstance(r, Exception) and is_engine_dead(r) for r in results):
                # Retrying is pointless and would mark every remaining row a hard
                # failure. Park them as pending so a rerun resumes cleanly.
                exc = next(r for r in results if isinstance(r, Exception) and is_engine_dead(r))
                print("\nInference engine died (see the vLLM traceback above, usually CUDA out of "
                      "memory). Stopping this phase; remaining rows are marked pending and will be "
                      "retried on the next run.")
                for c in batch + chunks_queue:
                    append_pending_rows(partial_pending, c, "engine_dead", repr(exc))
                pbar.update(sum(len(c) for c in batch + chunks_queue))
                chunks_queue.clear()
                break
            stop = False
            for current, res in zip(batch, results):
                if isinstance(res, TruncatedOutput):
                    handle_output(current, res.text, pre_exc=res)
                elif isinstance(res, Exception):
                    stop = handle_error(current, res) or stop
                else:
                    handle_output(current, res)
            if stop:
                break
    finally:
        pbar.close()
        if owned_generator:
            del generator
    print(
        f"Done. {len(raw_responses)} ok chunks, {len(hard_failures)} hard failures, "
        f"{len(chunks_queue)} chunks left queued."
    )

    # Clean & parse raw chunk responses into per-row records.
    parsed_records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    for item in raw_responses:
        try:
            parsed_records.extend(
                parse_chunk_response(
                    item["text"], item["row_ids"], spec.required_keys, id_of, text_of,
                    lenient=bool(item.get("lenient")),
                )
            )
        except Exception as exc:
            parse_failures.append({"row_ids": item["row_ids"], "error": repr(exc)})

    with open(log_path, "w", encoding="utf-8") as f:
        for entry in hard_failures:
            f.write(json.dumps({"kind": "api_failure", **entry}, ensure_ascii=False) + "\n")
        for entry in parse_failures:
            f.write(json.dumps({"kind": "parse_failure", **entry}, ensure_ascii=False) + "\n")

    # Final pending log: drop rows that later succeeded (self-heals stale entries).
    ok_row_ids = {rid for it in raw_responses for rid in it.get("row_ids", [])}
    pending_entries: list[dict[str, Any]] = []
    seen_pending: set[int] = set()
    if partial_pending.exists():
        with open(partial_pending, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                rid = entry.get("row_id")
                if rid in ok_row_ids or rid in seen_pending:
                    continue
                seen_pending.add(rid)
                pending_entries.append(entry)
    with open(pending_path, "w", encoding="utf-8") as f:
        for entry in pending_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    note_hit = sum(1 for r in parsed_records if str(r.get("NOTE", "")).strip())
    def _fixes(r: dict[str, Any]) -> list[str]:
        return [f for f in str(r.get("autofix", "")).split(";") if f]

    autofixed = sum(
        1 for r in parsed_records if any(not f.startswith("language_check") for f in _fixes(r))
    )
    lang_flags = sum(
        1 for r in parsed_records if any(f.startswith("language_check") for f in _fixes(r))
    )
    if autofixed:
        print(f"autofixed (deterministic codebook rules): {autofixed}/{len(parsed_records)} rows")
    reviewed = sum(1 for r in parsed_records if r.get("review_flags"))
    if reviewed:
        print(f"kept with review flags (see review_flags column): {reviewed}/{len(parsed_records)} rows")
    if lang_flags:
        print(
            f"LANGUAGE flagged for review (see autofix column): "
            f"{lang_flags}/{len(parsed_records)} rows"
        )
    print(
        f"parsed: {len(parsed_records)}/{len(df)} rows | "
        f"parse failures: {len(parse_failures)} | api failures: {len(hard_failures)} | "
        f"pending (transient): {len(pending_entries)} | "
        f"NOTE non-empty: {note_hit}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in parsed_records:
            record["model"] = model_id
            record["prompt_version"] = prompt_version
            record["precision"] = precision
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(parsed_records)} records -> {output_path}")

    res_df = pd.read_json(output_path, lines=True) if parsed_records else pd.DataFrame()
    if "_row_id" in res_df.columns:
        merged = df.merge(
            res_df, left_on="row_id", right_on="_row_id", how="left", suffixes=("", "_llm")
        )
        n_matched = int(merged["_row_id"].notna().sum())
        coverage = (n_matched / len(merged)) if len(merged) else 0.0
        merged["status"] = merged["_row_id"].notna().map({True: "coded", False: "missing"})
    else:
        merged = df.copy()
        merged["status"] = "missing"
        n_matched = 0
        coverage = 0.0
    print(f"Coverage: {coverage:.1%} ({n_matched}/{len(merged)} matched)")

    # Summary counts QA gate (generic: every non-key column is a coded field).
    print(f"n rows: {len(merged)}")
    for col in [c for c in merged.columns if c not in SKIP_SUMMARY_COLS]:
        if col == "model" or col.endswith("_llm"):
            continue
        if col in ("evidence", "evidence_en", "NOTE", "autofix", "review_flags"):
            continue
        series = (
            merged[col].astype(str)
            if merged[col].apply(lambda v: isinstance(v, list)).any()
            else merged[col]
        )
        print(f"\n== {col} ==")
        print(series.value_counts(dropna=False).head(20).to_string())

    keep = [c for c in spec.column_order if c in merged.columns]
    for drop_col in ("comment_id_llm", "video_id_llm", "text_llm"):
        if drop_col in merged.columns:
            merged = merged.drop(columns=[drop_col])
    for list_col in [c for c in keep if c not in ("row_id", "comment_id", "video_id", "text")]:
        if list_col in merged.columns:
            merged[list_col] = merged[list_col].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict))
                else v
            )
    # Uncoded rows make pandas upcast binary columns to float (0.0 / 1.0). Keep them
    # as nullable integers so the CSV holds 0 / 1 and blanks for uncoded rows.
    for c in BINARY_KEYS:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("Int64")
    merged.sort_values("row_id")[keep].to_csv(csv_path, index=False)
    print(f"Wrote {len(merged)} rows (coverage {coverage:.1%}) -> {csv_path}")
    clean = (
        len(hard_failures) == 0
        and len(parse_failures) == 0
        and len(pending_entries) == 0
        and n_matched == len(df)
    )
    if parsed_records:
        keys = ("id", "LANGUAGE", "NEG_SOCIAL_EMOTION", "FOCAL_EVENT", "CLUSTER", "CLUSTER_CONFLICT",
                "evidence", "confidence")
        print("\nSample record 0:", json.dumps({k: parsed_records[0].get(k) for k in keys}, ensure_ascii=False))
    print(
        f"\n{'=' * 60}\n"
        f"{'SUCCESS' if clean else 'FAILED'} [{phase}]: {len(parsed_records)}/{len(df)} rows coded | "
        f"{len(hard_failures)} hard failures | {len(parse_failures)} parse failures | "
        f"{len(pending_entries)} pending\n"
        + ("" if clean else f"Raw outputs of failures: {debug_path}\n")
        + f"{'=' * 60}"
    )
    return clean


def main(argv: Optional[list[str]] = None) -> int:
    """Run exactly the requested phases x passes. Returns process exit code."""
    args = parse_args(argv)
    phases = split_csv_arg(args.phases)
    pass_nums = [int(tok) for tok in split_csv_arg(args.passes)]
    for num in pass_nums:
        if num not in PASSES:
            print(f"Unknown pass: {num} (choose from {sorted(PASSES)})")
            return 2

    only_ids: Optional[set[str]] = None
    if args.only_ids is not None:
        if args.only_ids.suffix.lower() == ".csv":
            only_ids = set(pd.read_csv(args.only_ids, dtype=str)["comment_id"].str.strip())
        else:
            only_ids = {ln.strip() for ln in args.only_ids.read_text(encoding="utf-8").splitlines()
                        if ln.strip()}
        print(f"--only-ids: coding {len(only_ids)} listed comments only.")

    ok = True
    # One model load reused across all requested pairs (9B weights are slow to load).
    global LANGUAGE_RULE
    LANGUAGE_RULE = not args.no_language_rule
    precision = precision_label(args)
    print(f"Run precision: {precision}" + (f" | profile {args.gpu_profile}" if args.gpu_profile else ""))
    generator = LocalGenerator(
        args.model, args.max_new_tokens, args.load_in_4bit, args.max_model_len,
        low_memory=args.low_memory, gpu_memory_utilization=args.gpu_memory_utilization,
        concurrency=args.concurrency, min_new_tokens=args.min_new_tokens,
        fp8_weights=args.fp8_weights, fp8_kv_cache=args.fp8_kv_cache,
        compact_json=args.compact_json, attention_backend=args.attention_backend,
    )
    try:
        for phase in phases:
            df = load_phase_df(args.crisis, phase, args.limit)
            sample_info: Optional[dict[str, Any]] = None
            if only_ids is not None:
                # Keep row_ids from the full phase so ids line up across runs.
                df = df[df["comment_id"].isin(only_ids)]
                if df.empty:
                    print(f"Phase {phase}: no listed comments; skipped.")
                    continue
            elif args.sample_per_phase is not None:
                df, sample_info = select_sample(df, phase, args.sample_per_phase,
                                                args.sample_seed)
                print(f"Phase {phase}: sample {sample_info['n']} of {sample_info['N']} "
                      f"({sample_info['design']})")
            print(f"Phase {phase}: {len(df)} rows")
            for num in pass_nums:
                spec = PASSES[num]
                print(f"=== Local LLM [{args.crisis}]: {phase} / pass{num} ===")
                clean = run_pair(
                    df,
                    spec,
                    args.crisis,
                    phase,
                    args.model,
                    args.chunk_size,
                    args.max_new_tokens,
                    args.overwrite,
                    generator,
                    args.closing,
                    concurrency=max(1, args.concurrency),
                    run_name=args.run_name,
                    sample_info=sample_info,
                    precision=precision,
                )
                ok = ok and clean
    finally:
        del generator
    print(f"\nOverall: {'ALL PAIRS CLEAN' if ok else 'ONE OR MORE PAIRS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())