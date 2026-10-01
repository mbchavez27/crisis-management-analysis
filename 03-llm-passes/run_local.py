"""Local-LLM runner for the v2 full-codebook coding pass.

Mirrors the full-codebook-coding notebook cell-for-cell except the micro-sample
step: full phase CSVs are coded instead of head(n_sample). Passes 1-3
(notebooks and prompts) are archived under legacy/; this runner serves the
single active v2 codebook pass.

Opt-in execution: --phases and --passes are both required, and the script
runs exactly their cartesian product (one phase x one prompt per run pair).

Example:
    uv run python run_local.py --phases 04-reckoning --passes 1
    uv run python run_local.py --phases 04-reckoning --passes 1 --limit 10
    uv run python run_local.py --phases 04-reckoning --passes 1 --load-in-4bit

--load-in-4bit: nf4 quantization for <=12GB GPUs. Full precision is the
default so canonical runs (GPU VM) stay byte-identical across machines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).parent
PROMPTS_DIR = HERE / "prompts"
OUTPUT_ROOT = HERE / "output"

DEFAULT_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B"
SKIP_SUMMARY_COLS = {"row_id", "comment_id", "video_id", "text", "id", "_row_id", "row_ids"}

# Outage-resilient retry: exponential backoff, requeue, circuit breaker.
# Values mirror the full-codebook-coding notebook inference cell exactly.
TRANSIENT_TRIES = 5       # 5 attempts => transient sleeps ~10/20/40/80s
TRANSIENT_BASE_S = 10
BACKOFF_CAP_S = 300
RATE_LIMIT_SLEEP = (60, 90)
COOLDOWN_S = 60           # pause before a requeued chunk is attempted again
MAX_REVISITS = 3          # requeues per chunk before it is marked pending
BREAKER_TRIP = 5          # consecutive outage-exhausted chunks before abort

# Restates the task after the comments so the output spec is the last thing read.
CLOSING = (
    "\n\nNow output exactly one JSON object per comment above, one per line, "
    "following the OUTPUT format. Output only JSON, no other text."
)


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


def generate_with_retry(
    generator: Any, prompt: str, tries: int = TRANSIENT_TRIES, **kw: Any
) -> str:
    """Generate one chunk with exponential backoff for outages (mirrors the notebook)."""
    for attempt in range(tries):
        try:
            return generator.generate(prompt, **kw)
        except Exception as e:
            if attempt == tries - 1:
                raise
            if is_rate_limited(e):
                sleep_time = random.uniform(*RATE_LIMIT_SLEEP)
            elif is_transient(e):
                sleep_time = min(TRANSIENT_BASE_S * (2 ** attempt) + random.uniform(0, 5), BACKOFF_CAP_S)
            else:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"  [Attempt {attempt+1} failed: {repr(e)}] Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)


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
    {
        "id",
        "LANGUAGE",
        "ANGER",
        "CONTEMPT",
        "DISGUST",
        "SADNESS",
        "CONDOLENCE",
        "COMPASSION_VICTIMS",
        "SYMPATHY_OFFENDER",
        "FEAR",
        "NORM_VIOLATION",
        "ATTRIBUTION",
        "IGNORED_WARNING",
        "RECURRENCE",
        "COVER_UP",
        "ALT_ACCOUNT",
        "INJUSTICE",
        "SELF_INTEREST",
        "BLAME_TARGETS",
        "STANCE",
        "NEEDS_CONTEXT",
        "VALUES",
        "CLASS_FRAME",
        "CLASS_DIRECTION",
        "IDENTITY_FRAME",
        "APPEARANCE_ATTACK",
        "ACTIONS",
        "RESPONSE_TARGET",
        "RESPONSE_EVAL",
        "REJECT_REASONS",
        "TARGET_OF_COMMENT",
        "META_DISCOURSE",
        "SARCASM",
        "UNCODABLE",
        "evidence",
        "evidence_en",
        "EMERGENT",
        "OTHER_NOTE",
        "confidence",
    }
)

COLUMN_ORDER = [
    "row_id",
    "comment_id",
    "video_id",
    "text",
    "id",
    "LANGUAGE",
    "ANGER",
    "CONTEMPT",
    "DISGUST",
    "SADNESS",
    "CONDOLENCE",
    "COMPASSION_VICTIMS",
    "SYMPATHY_OFFENDER",
    "FEAR",
    "NORM_VIOLATION",
    "ATTRIBUTION",
    "IGNORED_WARNING",
    "RECURRENCE",
    "COVER_UP",
    "ALT_ACCOUNT",
    "INJUSTICE",
    "SELF_INTEREST",
    "BLAME_TARGETS",
    "STANCE",
    "NEEDS_CONTEXT",
    "VALUES",
    "CLASS_FRAME",
    "CLASS_DIRECTION",
    "IDENTITY_FRAME",
    "APPEARANCE_ATTACK",
    "ACTIONS",
    "RESPONSE_TARGET",
    "RESPONSE_EVAL",
    "REJECT_REASONS",
    "TARGET_OF_COMMENT",
    "META_DISCOURSE",
    "SARCASM",
    "UNCODABLE",
    "evidence",
    "evidence_en",
    "EMERGENT",
    "OTHER_NOTE",
    "confidence",
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
        outdir="full-codebook-coding",
        prefix="full_codebook",
        required_keys=REQUIRED_KEYS,
        column_order=COLUMN_ORDER,
    ),
}


def build_codebook_prompt(crisis: str, phase: str) -> str:
    """Assemble the v2 prompt: wrapper + codebook text + per-phase context.

    Mirrors the full-codebook-coding notebook's prompt-assembly cell so
    notebook and runner send byte-identical instructions for the same phase.
    """
    codebook = (PROMPTS_DIR / "codebook-v2.1.md").read_text(encoding="utf-8").strip()
    wrapper = (PROMPTS_DIR / "coding-prompt.txt").read_text(encoding="utf-8").strip()
    contexts = json.loads((PROMPTS_DIR / "phase-context.json").read_text(encoding="utf-8"))
    case_context = contexts.get(crisis, {}).get(phase, "")
    if not case_context:
        raise RuntimeError(f"No phase context for {crisis}/{phase} in phase-context.json.")
    prompt = (
        wrapper.replace("MODE: [PILOT | FULL]", "MODE: FULL")
        .replace("<<<paste full text of the codebook here>>>", codebook)
        .replace(
            "[The data are made about a Youtube video on ..... No judgments about fault.]",
            case_context,
        )
    )
    for marker in ("<<<paste", "PILOT | FULL", "....."):
        if marker in prompt:
            raise RuntimeError(f"Prompt assembly missed placeholder: {marker}")
    return prompt


def to_line(cid: Any, txt: Any) -> str:
    """Format one comment as an `id | text` line (newlines flattened)."""
    flat = str(txt).replace("\r", " ").replace("\n", " ").strip()
    return f"{cid} | {flat}"


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
        "--max-new-tokens", type=int, default=8192, help="Generation ceiling per chunk."
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="nf4 quantization for <=12GB GPUs (~6GB for the 9B default). Full precision otherwise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard partial checkpoints and start fresh.",
    )
    return parser.parse_args(argv)


def split_csv_arg(value: str) -> list[str]:
    """Split a comma-separated flag into cleaned tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def load_phase_df(crisis: str, phase: str, limit: Optional[int]) -> pd.DataFrame:
    """Load a full phase CSV in file order with row_ids for lossless merging."""
    input_path = (
        HERE / ".." / "01-preprocessing" / "output" / crisis / "youtube" / f"{phase}.csv"
    ).resolve()
    full_df = pd.read_csv(input_path)
    if limit is not None:
        full_df = full_df.head(limit)
    df = full_df[["comment_id", "video_id", "text"]].copy()
    df = df.dropna()
    for col in ("comment_id", "video_id", "text"):
        df[col] = df[col].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
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


def parse_chunk_response(
    raw: str,
    row_ids: list[int],
    required: frozenset[str],
    id_of: Optional[dict[int, str]] = None,
) -> list[dict[str, Any]]:
    """Parse one chunk response into per-row records.

    Raises ValueError on empty output, bad JSON, count mismatch, missing
    keys, or id mismatch so the caller can split-and-retry or log a hard
    failure.
    """
    if not raw.strip():
        raise ValueError("empty_response")
    try:
        objs = load_objs(raw)
    except Exception as exc:
        raise ValueError(f"json_decode: {exc!r}") from exc
    if isinstance(objs, dict):
        objs = [objs]
    if len(objs) > len(row_ids):
        objs = objs[: len(row_ids)]  # model kept going; the id check guards correctness
    if len(objs) != len(row_ids):
        raise ValueError(f"count_mismatch: {len(objs)} objs for {len(row_ids)} rows")
    records: list[dict[str, Any]] = []
    for rid, obj in zip(row_ids, objs):
        missing = required - set(obj.keys())
        if missing:
            raise ValueError(f"missing_keys: {sorted(missing)}")
        if id_of is not None:
            want = str(id_of.get(rid, "")).strip()
            got = str(obj.get("id", "")).strip()
            if got != want:
                raise ValueError(f"id_mismatch: {got!r} vs {want!r}")
        obj["_row_id"] = rid
        records.append(obj)
    return records


class LocalGenerator:
    """Thin wrapper around a transformers causal LM. Imported lazily."""

    def __init__(self, model_id: str, max_new_tokens: int, load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self._printed_len = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            try:
                dtype = torch.bfloat16
                # Validate bf16 support before committing to it.
                torch.zeros(1, dtype=dtype, device="cuda")
            except Exception:
                dtype = torch.float16
        else:
            dtype = torch.float32

        # 4-bit nf4 for <=12GB GPUs; full precision stays the default so
        # canonical (A100) runs are byte-identical to pre-quantization output.
        if load_in_4bit:
            if not has_cuda:
                raise RuntimeError("--load-in-4bit requires a CUDA GPU.")
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, quantization_config=quant_config, device_map="auto"
            )
            precision = "4-bit nf4"
        # device_map="auto" needs accelerate; otherwise pin to one device.
        # Local 9B weights expect a GPU for production throughput; CPU works but slowly.
        elif has_accelerate:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map="auto"
            )
            precision = str(dtype).removeprefix("torch.")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            self.model.to("cuda" if has_cuda else "cpu")
            precision = str(dtype).removeprefix("torch.")
        self.model.eval()
        print(f"Model [{model_id}] loaded in {precision} on {'cuda' if has_cuda else 'cpu'}.")

    def generate(self, prompt: str, prefill: str = "") -> str:
        """Deterministic generation. `prefill` is forced as the start of the reply."""
        import torch

        try:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            has_template = True
        except Exception:
            chat, has_template = prompt, False
        chat += prefill
        # The chat template already contains BOS; don't add a second one.
        inputs = self.tokenizer(
            chat, return_tensors="pt", add_special_tokens=not has_template
        ).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        if not self._printed_len:
            print(f"prompt tokens: {n_in}")
            self._printed_len = True
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = out[0][n_in:]
        if len(gen) >= self.max_new_tokens:
            print(f"  [warn] hit max_new_tokens ({self.max_new_tokens}); output may be truncated")
        return prefill + self.tokenizer.decode(gen, skip_special_tokens=True)


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
) -> bool:
    """Code one phase x one pass with checkpoint resume. Returns True if clean."""
    pass_dir = OUTPUT_ROOT / crisis / spec.outdir
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
    id_of = dict(zip(df["row_id"].tolist(), df["comment_id"].astype(str).tolist()))

    chunks_queue = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    raw_responses: list[dict[str, Any]] = []
    hard_failures: list[dict[str, Any]] = []
    revisit_counts: dict[tuple[int, ...], int] = {}
    outage_streak = 0

    # Resume: successes filter the queue; prior failures are retried, never skipped.
    if not overwrite:
        try:
            ok_ids: set[int] = set()
            if partial_resp.exists():
                with open(partial_resp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            ok_ids.update(rec.get("row_ids", []))
                            raw_responses.append(rec)
                        except Exception:
                            pass
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
        while chunks_queue:
            current = chunks_queue.pop(0)
            n_rows = len(current)
            row_ids = current["row_id"].tolist()
            payload = "\n".join(
                to_line(r["comment_id"], r["text"]) for _, r in current.iterrows()
            )
            # Force the reply to start as the first comment's JSON object.
            prefill = '{"id":"' + str(current.iloc[0]["comment_id"]) + '",'
            try:
                raw = generate_with_retry(
                    generator,
                    prompt_template + "\n\nCOMMENTS TO CODE\n" + payload + CLOSING,
                    prefill=prefill,
                )
            except Exception as exc:
                if is_rate_limited(exc) or is_transient(exc):
                    reason = (
                        "rate_limited_exhausted"
                        if is_rate_limited(exc)
                        else "transient_exhausted"
                    )
                    outage_streak += 1
                    key = tuple(row_ids)
                    revisit_counts[key] = revisit_counts.get(key, 0) + 1
                    if revisit_counts[key] > MAX_REVISITS:
                        print(
                            f"   -> {reason} after {MAX_REVISITS} requeues. "
                            f"Marking {n_rows} rows pending."
                        )
                        append_pending_rows(partial_pending, current, reason, repr(exc))
                        pbar.update(n_rows)
                    else:
                        print(
                            f"   -> {reason}. Requeueing chunk "
                            f"({revisit_counts[key]}/{MAX_REVISITS}); "
                            f"cooldown {COOLDOWN_S}s."
                        )
                        time.sleep(COOLDOWN_S)
                        chunks_queue.append(current)
                    if outage_streak >= BREAKER_TRIP:
                        print(
                            f"Circuit breaker: {outage_streak} consecutive outage-"
                            f"exhausted chunks. Aborting with {len(chunks_queue)} "
                            f"chunks queued — partials saved, rerun to resume."
                        )
                        for c in chunks_queue:
                            append_pending_rows(partial_pending, c, "breaker_abort", repr(exc))
                        break
                    continue
                split_or_log(current, n_rows, exc)
                continue
            # Validate now so bad chunks split immediately instead of at export.
            try:
                parse_chunk_response(raw or "", row_ids, spec.required_keys, id_of)
            except Exception as exc:
                dump_raw(debug_path, row_ids, repr(exc), raw or "")
                split_or_log(current, n_rows, exc)
                continue
            rec = {"row_ids": row_ids, "text": raw or ""}
            raw_responses.append(rec)
            with open(partial_resp, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outage_streak = 0
            pbar.update(n_rows)
            time.sleep(0.1)
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
                    item["text"], item["row_ids"], spec.required_keys, id_of
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

    emergent_hit = sum(1 for r in parsed_records if str(r.get("EMERGENT", "")).strip())
    print(
        f"parsed: {len(parsed_records)}/{len(df)} rows | "
        f"parse failures: {len(parse_failures)} | api failures: {len(hard_failures)} | "
        f"pending (transient): {len(pending_entries)} | "
        f"EMERGENT non-empty (FULL-mode warning): {emergent_hit}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in parsed_records:
            record["model"] = model_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(parsed_records)} records -> {output_path}")

    res_df = pd.read_json(output_path, lines=True) if parsed_records else pd.DataFrame()
    if "_row_id" in res_df.columns:
        merged = df.merge(
            res_df, left_on="row_id", right_on="_row_id", how="left", suffixes=("", "_llm")
        )
        n_matched = int(merged["_row_id"].notna().sum())
        coverage = (n_matched / len(merged)) if len(merged) else 0.0
    else:
        merged = df.copy()
        n_matched = 0
        coverage = 0.0
    print(f"Coverage: {coverage:.1%} ({n_matched}/{len(merged)} matched)")

    # Summary counts QA gate (generic: every non-key column is a coded field).
    print(f"n rows: {len(merged)}")
    for col in [c for c in merged.columns if c not in SKIP_SUMMARY_COLS]:
        if col == "model" or col.endswith("_llm"):
            continue
        if col in ("evidence", "evidence_en", "EMERGENT", "OTHER_NOTE"):
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
    merged.sort_values("row_id")[keep].to_csv(csv_path, index=False)
    print(f"Wrote {len(merged)} rows (coverage {coverage:.1%}) -> {csv_path}")
    return (
        len(hard_failures) == 0
        and len(parse_failures) == 0
        and len(pending_entries) == 0
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Run exactly the requested phases x passes. Returns process exit code."""
    args = parse_args(argv)
    phases = split_csv_arg(args.phases)
    pass_nums = [int(tok) for tok in split_csv_arg(args.passes)]
    for num in pass_nums:
        if num not in PASSES:
            print(f"Unknown pass: {num} (choose from {sorted(PASSES)})")
            return 2

    ok = True
    # One model load reused across all requested pairs (9B weights are slow to load).
    generator = LocalGenerator(args.model, args.max_new_tokens, args.load_in_4bit)
    try:
        for phase in phases:
            df = load_phase_df(args.crisis, phase, args.limit)
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
                )
                ok = ok and clean
    finally:
        del generator
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
