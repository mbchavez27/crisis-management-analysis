"""Local-LLM runner for the 03-llm-passes coding passes.

Mirrors the pass notebooks cell-for-cell except the micro-sample step:
full phase CSVs are coded instead of head(n_sample). One file serves all
three passes; per-pass differences (prompt, output dir, JSON schema) live
in the PASSES table so shared logic is never triplicated.

Opt-in execution: --phases and --passes are both required, and the script
runs exactly their cartesian product (one phase x one prompt per run pair).

Example:
    uv run python run_local.py --phases 04-reckoning --passes 1
    uv run python run_local.py --phases 04-reckoning --passes 1,2,3 --limit 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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
SKIP_SUMMARY_COLS = {"row_id", "video_id", "text", "_row_id", "row_ids"}


@dataclass(frozen=True)
class PassSpec:
    """Per-pass configuration. Everything else is shared logic."""

    num: int
    prompt_file: str
    outdir: str
    prefix: str
    required_keys: frozenset[str]
    column_order: list[str]


PASSES: dict[int, PassSpec] = {
    1: PassSpec(
        num=1,
        prompt_file="pass-01-grief-outrage.txt",
        outdir="pass-01-grief-outrage",
        prefix="pass01",
        required_keys=frozenset(
            {
                "video_id",
                "text",
                "grief_or_outrage",
                "value_or_relationship_invoked",
                "family_referenced",
            }
        ),
        column_order=[
            "row_id",
            "video_id",
            "text",
            "grief_or_outrage",
            "value_or_relationship_invoked",
            "family_referenced",
        ],
    ),
    2: PassSpec(
        num=2,
        prompt_file="pass-02-basic-emotions.txt",
        outdir="pass-02-basic-emotions",
        prefix="pass02",
        required_keys=frozenset(
            {
                "video_id",
                "text",
                "basic_emotions",
                "value_or_relationship_invoked",
                "family_referenced",
            }
        ),
        column_order=[
            "row_id",
            "video_id",
            "text",
            "basic_emotions",
            "value_or_relationship_invoked",
            "family_referenced",
        ],
    ),
    3: PassSpec(
        num=3,
        prompt_file="pass-03-actions-demanded.txt",
        outdir="pass-03-actions-demanded",
        prefix="pass03",
        required_keys=frozenset({"video_id", "text", "action_demanded"}),
        column_order=["row_id", "video_id", "text", "action_demanded"],
    ),
}


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
        help="Comma-separated pass numbers from {1,2,3}, e.g. '1' or '1,2,3'.",
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
        "--max-new-tokens", type=int, default=2048, help="Generation ceiling per chunk."
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
    df = full_df[["video_id", "text"]].copy()
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["row_id"] = df.index
    return df


def strip_fences(raw: str) -> str:
    """Remove a single wrapping markdown code fence from model output."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_chunk_response(
    raw: str, row_ids: list[int], required: frozenset[str]
) -> list[dict[str, Any]]:
    """Parse one chunk response into per-row records.

    Raises ValueError on empty output, bad JSON, count mismatch, or
    missing keys so the caller can split-and-retry or log a hard failure.
    """
    if not raw.strip():
        raise ValueError("empty_response")
    try:
        objs = json.loads(strip_fences(raw))
    except Exception as exc:
        raise ValueError(f"json_decode: {exc!r}") from exc
    if isinstance(objs, dict):
        objs = [objs]
    if len(objs) != len(row_ids):
        raise ValueError(f"count_mismatch: {len(objs)} objs for {len(row_ids)} rows")
    records: list[dict[str, Any]] = []
    for rid, obj in zip(row_ids, objs):
        missing = required - set(obj.keys())
        if missing:
            raise ValueError(f"missing_keys: {sorted(missing)}")
        obj["_row_id"] = rid
        records.append(obj)
    return records


class LocalGenerator:
    """Thin wrapper around a transformers causal LM. Imported lazily."""

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
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

        # device_map="auto" needs accelerate; otherwise pin to one device.
        # Local 9B weights expect a GPU for production throughput; CPU works but slowly.
        if has_accelerate:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            self.model.to("cuda" if has_cuda else "cpu")
        self.model.eval()

    def generate(self, prompt: str) -> str:
        """Generate a response for one chunk prompt (deterministic decoding)."""
        import torch

        try:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            chat = prompt
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


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
    csv_path = pass_dir / f"{spec.prefix}_results.csv"
    partial_resp = Path(str(output_path) + ".partial.jsonl")
    partial_fail = Path(str(log_path) + ".partial.log")

    prompt_template = (PROMPTS_DIR / spec.prompt_file).read_text(encoding="utf-8").strip()

    chunks_queue = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    raw_responses: list[dict[str, Any]] = []
    hard_failures: list[dict[str, Any]] = []

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
                    f"{len(chunks_queue)} chunks left."
                )
        except Exception as exc:
            print(f"Resume check failed ({exc!r}); starting fresh.")
    else:
        for p in (partial_resp, partial_fail):
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
    try:
        while chunks_queue:
            current = chunks_queue.pop(0)
            n_rows = len(current)
            row_ids = current["row_id"].tolist()
            csv_str = current[["video_id", "text"]].to_csv(index=False)
            try:
                raw = generator.generate(prompt_template + "\n\nData:\n" + csv_str)
                rec = {"row_ids": row_ids, "text": raw or ""}
                # Validate now so bad chunks split immediately instead of at export.
                parse_chunk_response(raw or "", row_ids, spec.required_keys)
                raw_responses.append(rec)
                with open(partial_resp, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.update(n_rows)
            except Exception as exc:
                print(f"Chunk failure (size {n_rows}): {exc!r}")
                if n_rows > 1:
                    mid = n_rows // 2
                    chunks_queue.insert(0, current.iloc[mid:])
                    chunks_queue.insert(0, current.iloc[:mid])
                else:
                    row = current.iloc[0]
                    entry = {
                        "row_id": int(row["row_id"]),
                        "video_id": row.get("video_id"),
                        "reason": "content_failure",
                        "error": repr(exc),
                        "text_length": len(str(row["text"])),
                    }
                    hard_failures.append(entry)
                    with open(partial_fail, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps({"kind": "api_failure", **entry}, ensure_ascii=False)
                            + "\n"
                        )
                    pbar.update(1)
            time.sleep(0.1)
    finally:
        pbar.close()
        if owned_generator:
            del generator
    print(f"Done. {len(raw_responses)} ok chunks, {len(hard_failures)} hard failures.")

    # Clean & parse raw chunk responses into per-row records.
    parsed_records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    for item in raw_responses:
        try:
            parsed_records.extend(
                parse_chunk_response(
                    item["text"], item["row_ids"], spec.required_keys
                )
            )
        except Exception as exc:
            parse_failures.append({"row_ids": item["row_ids"], "error": repr(exc)})

    with open(log_path, "w", encoding="utf-8") as f:
        for entry in hard_failures:
            f.write(json.dumps({"kind": "api_failure", **entry}, ensure_ascii=False) + "\n")
        for entry in parse_failures:
            f.write(json.dumps({"kind": "parse_failure", **entry}, ensure_ascii=False) + "\n")
    print(
        f"parsed: {len(parsed_records)}/{len(df)} rows | "
        f"parse failures: {len(parse_failures)} | api failures: {len(hard_failures)}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in parsed_records:
            record["model"] = model_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(parsed_records)} records -> {output_path}")

    res_df = pd.read_json(output_path, lines=True) if parsed_records else pd.DataFrame()
    merged = df.merge(
        res_df, left_on="row_id", right_on="_row_id", how="left", suffixes=("", "_llm")
    )
    coverage = merged["_row_id"].notna().mean() if len(merged) else 0.0
    print(f"Coverage: {coverage:.1%} ({merged['_row_id'].notna().sum()}/{len(merged)} matched)")

    # Summary counts QA gate (generic: every non-key column is a coded field).
    print(f"n rows: {len(merged)}")
    for col in [c for c in merged.columns if c not in SKIP_SUMMARY_COLS]:
        if col == "model" or col.endswith("_llm"):
            continue
        series = (
            merged[col].astype(str)
            if merged[col].apply(lambda v: isinstance(v, list)).any()
            else merged[col]
        )
        print(f"\n== {col} ==")
        print(series.value_counts(dropna=False).head(20).to_string())

    keep = [c for c in spec.column_order if c in merged.columns]
    for drop_col in ("video_id_llm", "text_llm"):
        if drop_col in merged.columns:
            merged = merged.drop(columns=[drop_col])
    for list_col in [c for c in keep if c not in ("row_id", "video_id", "text")]:
        if list_col in merged.columns:
            merged[list_col] = merged[list_col].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict))
                else v
            )
    merged.sort_values("row_id")[keep].to_csv(csv_path, index=False)
    print(f"Wrote {len(merged)} rows (coverage {coverage:.1%}) -> {csv_path}")
    return len(hard_failures) == 0 and len(parse_failures) == 0


def main(argv: Optional[list[str]] = None) -> int:
    """Run exactly the requested phases x passes. Returns process exit code."""
    args = parse_args(argv)
    phases = split_csv_arg(args.phases)
    pass_nums = [int(tok) for tok in split_csv_arg(args.passes)]
    for num in pass_nums:
        if num not in PASSES:
            print(f"Unknown pass: {num} (choose from 1,2,3)")
            return 2

    ok = True
    # One model load reused across all requested pairs (9B weights are slow to load).
    generator = LocalGenerator(args.model, args.max_new_tokens)
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
