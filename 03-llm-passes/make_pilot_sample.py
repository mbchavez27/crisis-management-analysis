#!/usr/bin/env python3
"""Draw the human-validation (pilot) sample for the full-codebook pass.

Two strata, drawn from the preprocessed source comments (not the LLM output, so
comments the model failed on are included):

  random  - a simple random sample, allocated across phases in proportion to their
            size. Estimates LLM-human agreement for every field as it occurs.
  class   - comments containing class vocabulary (F1: wealth, poverty, privilege,
            scholarship status, rich vs poor ...), excluding those already drawn.
            CLASS_FRAME is rare (~4%), so a random 200 holds too few class comments
            to measure agreement on the construct the RQs depend on.

Outputs (in 03-llm-passes/pilot/{crisis}/):
  pilot_sample.csv        coding sheet: pilot_id, stratum, phase, ids, text, sampling
                          info, then blank code columns for human coders (blind: no LLM
                          codes in it)
  pilot_comment_ids.txt   for: run_local.py --only-ids pilot/{crisis}/pilot_comment_ids.txt
  pilot_design.json       seed, sizes, pattern and per-phase counts (for the methods)

Weights: `weight` is the inverse selection probability within the stratum and phase
(N_stratum_phase / n_stratum_phase). Agreement on the random stratum is unweighted
apart from the proportional allocation; the class stratum is an oversample and is
reported separately (or weighted) - never pooled unweighted with the random stratum.

Usage:
    uv run --no-sync python make_pilot_sample.py                       # 150 random + 75 class
    uv run --no-sync python make_pilot_sample.py --n-random 200 --n-class 100 --seed 11
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

CLASS_WORDS = (
    r"\b(?:mahirap|mahihirap|mayaman|mayayaman|elite|elitist|privileged?|privilege|nepo\w*|"
    r"scholar\w*|iskolar|pera|money|millions?|billions?|rich|poor|probinsya\w*|"
    r"working class|middle class|tuition|matrikula|afford\w*|breadwinner|classism|burgis|"
    r"connections?|palakasan|kapit|kahirapan|dukha|maralita|sosyal|coñ?o)\b"
)

# Human coding columns, in codebook order (all 33 codes + LANGUAGE).
CODE_COLUMNS = [
    "LANGUAGE",
    "ANGER", "CONTEMPT", "DISGUST", "SADNESS", "CONDOLENCE", "COMPASSION_VICTIMS",
    "SYMPATHY_OFFENDER", "FEAR",
    "NORM_VIOLATION", "ATTRIBUTION", "IGNORED_WARNING", "RECURRENCE", "COVER_UP",
    "ALT_ACCOUNT", "INJUSTICE", "SELF_INTEREST", "BLAME_TARGETS",
    "STANCE", "NEEDS_CONTEXT",
    "VALUES",
    "CLASS_FRAME", "CLASS_DIRECTION", "IDENTITY_FRAME", "APPEARANCE_ATTACK",
    "ACTIONS",
    "RESPONSE_TARGET", "RESPONSE_EVAL", "REJECT_REASONS",
    "TARGET_OF_COMMENT", "META_DISCOURSE", "SARCASM", "UNCODABLE",
]


def load_source(crisis: str, phases: list[str] | None) -> pd.DataFrame:
    """Load preprocessed comments exactly as run_local.load_phase_df does."""
    src = HERE.parent / "01-preprocessing" / "output" / crisis / "youtube"
    files = sorted(src.glob("*.csv"))
    if phases:
        files = [f for f in files if f.stem in phases]
    frames = []
    for f in files:
        d = pd.read_csv(f)[["comment_id", "video_id", "text"]].dropna()
        for c in ("comment_id", "video_id", "text"):
            d[c] = d[c].astype(str).str.strip()
        d = d[d["text"] != ""].drop_duplicates("comment_id", keep="first").reset_index(drop=True)
        d["row_id"] = d.index  # same row_id as run_local.py
        d["phase"] = f.stem
        frames.append(d)
    if not frames:
        raise SystemExit(f"No source CSVs found in {src}")
    return pd.concat(frames, ignore_index=True)


def allocate(sizes: pd.Series, n: int, min_per: int) -> pd.Series:
    """Proportional allocation with a floor, summing exactly to n (largest remainder)."""
    n = min(n, int(sizes.sum()))
    floor = np.minimum(sizes, min_per)
    rest = n - int(floor.sum())
    if rest < 0:
        raise SystemExit(f"--min-per-phase too large for n={n}")
    room = sizes - floor
    share = room / room.sum() * rest if room.sum() else room * 0
    alloc = floor + np.floor(share).astype(int)
    for p in (share - np.floor(share)).sort_values(ascending=False).index:
        if alloc.sum() >= n:
            break
        if alloc[p] < sizes[p]:
            alloc[p] += 1
    return alloc.astype(int)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crisis", default="ateneo-drowning")
    ap.add_argument("--phases", default="all", help="'all' or comma-separated phase slugs")
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--n-class", type=int, default=75)
    ap.add_argument("--min-per-phase", type=int, default=10, help="floor per phase in each stratum")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    phases = None if args.phases.strip().lower() == "all" else [p.strip() for p in args.phases.split(",")]
    df = load_source(args.crisis, phases)
    rng = np.random.default_rng(args.seed)
    df["class_vocab"] = df["text"].str.contains(CLASS_WORDS, flags=re.I, regex=True)

    # Random stratum: proportional to phase size.
    sizes = df.groupby("phase").size()
    alloc_r = allocate(sizes, args.n_random, args.min_per_phase)
    rand = []
    for p, k in alloc_r.items():
        pool = df[df["phase"] == p]
        pick = pool.sample(n=int(k), random_state=int(rng.integers(1 << 31)))
        rand.append(pick.assign(stratum="random", N_stratum_phase=len(pool), n_stratum_phase=int(k)))
    rand = pd.concat(rand)

    # Class stratum: class-vocabulary comments not already drawn.
    cls_pool = df[df["class_vocab"] & ~df.index.isin(rand.index)]
    cls = pd.DataFrame()
    if args.n_class > 0 and len(cls_pool):
        csizes = cls_pool.groupby("phase").size()
        alloc_c = allocate(csizes, args.n_class, min(args.min_per_phase, int(csizes.min())))
        parts = []
        for p, k in alloc_c.items():
            pool = cls_pool[cls_pool["phase"] == p]
            pick = pool.sample(n=int(k), random_state=int(rng.integers(1 << 31)))
            parts.append(pick.assign(stratum="class", N_stratum_phase=len(pool), n_stratum_phase=int(k)))
        cls = pd.concat(parts)

    sample = pd.concat([rand, cls]).sample(frac=1, random_state=args.seed)  # shuffle: blind to stratum order
    sample["weight"] = (sample["N_stratum_phase"] / sample["n_stratum_phase"]).round(3)
    sample.insert(0, "pilot_id", [f"P{i:04d}" for i in range(1, len(sample) + 1)])
    sheet = sample[["pilot_id", "stratum", "phase", "comment_id", "video_id", "row_id", "text",
                    "class_vocab", "N_stratum_phase", "n_stratum_phase", "weight"]].copy()
    for c in CODE_COLUMNS:
        sheet[c] = ""
    sheet["coder"] = ""
    sheet["notes"] = ""

    out = args.out_dir or (HERE / "pilot" / args.crisis)
    out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out / "pilot_sample.csv", index=False, encoding="utf-8-sig")  # BOM: opens cleanly in Excel
    (out / "pilot_comment_ids.txt").write_text("\n".join(sheet["comment_id"]) + "\n", encoding="utf-8")
    design = {
        "crisis": args.crisis, "seed": args.seed, "n_random": int(len(rand)), "n_class": int(len(cls)),
        "min_per_phase": args.min_per_phase, "class_pattern": CLASS_WORDS,
        "phase_sizes": {k: int(v) for k, v in sizes.items()},
        "class_vocab_sizes": {k: int(v) for k, v in df[df["class_vocab"]].groupby("phase").size().items()},
        "random_per_phase": {k: int(v) for k, v in alloc_r.items()},
        "class_per_phase": ({k: int(v) for k, v in cls.groupby("phase").size().items()} if len(cls) else {}),
        "class_vocab_in_random_stratum": int(rand["class_vocab"].sum()),
    }
    (out / "pilot_design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pilot sample: {len(rand)} random + {len(cls)} class = {len(sheet)} comments")
    print(pd.crosstab(sheet["phase"], sheet["stratum"], margins=True).to_string())
    print(f"class-vocabulary comments in the random stratum: {design['class_vocab_in_random_stratum']}")
    print(f"\nCoding sheet  -> {out / 'pilot_sample.csv'}")
    print(f"Comment ids   -> {out / 'pilot_comment_ids.txt'}")
    print(f"Design record -> {out / 'pilot_design.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
