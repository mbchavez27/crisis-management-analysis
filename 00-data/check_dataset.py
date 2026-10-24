#!/usr/bin/env python3
"""Check that a crisis dataset under 00-data/{slug}/ matches the format the pipeline reads.

Standard library only, so it runs before any stage environment is set up.

Usage (from the repo root):
    python3 00-data/check_dataset.py ateneo-drowning
    python3 00-data/check_dataset.py my-crisis --quiet      # only problems and the summary

Exit code 0 = no errors (warnings are allowed), 1 = at least one error.
The format itself is documented in 00-data/README.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DATA_ROOT.parent

PHASES = ["01-onset", "02-confirmation", "03-spread", "04-reckoning", "05-lingering"]
REDDIT_EXTRA_PHASES = ["00-unrelated"]

# Columns the 01-preprocessing notebooks read. Extra columns are fine.
YOUTUBE_REQUIRED = ["video_id", "comment_id", "parent_comment_id", "text",
                    "like_count", "create_time", "user_handle"]
REDDIT_REQUIRED = ["type", "id", "author", "author_fullname", "replying_to_user",
                   "metadata_json", "score", "created_time", "text", "subreddit"]

VTT_CUE = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n")
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class Report:
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.errors = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        if not self.quiet:
            print(f"  ok    {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  WARN  {msg}")

    def error(self, msg: str) -> None:
        self.errors += 1
        print(f"  ERROR {msg}")


def read_csv(path: Path, rep: Report) -> tuple[list[str], list[dict]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    except UnicodeDecodeError:
        rep.error(f"{path.relative_to(DATA_ROOT)}: not UTF-8. Re-save it as UTF-8")
    except csv.Error as e:
        rep.error(f"{path.relative_to(DATA_ROOT)}: unreadable CSV ({e})")
    return None


def is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def check_youtube(crisis_dir: Path, rep: Report) -> list[str]:
    """Return the phases that have YouTube comments."""
    print("\nYouTube comments (required)")
    yt = crisis_dir / "youtube-comments"
    if not yt.is_dir():
        rep.error("youtube-comments/ is missing. Every crisis needs YouTube comments")
        return []

    for d in sorted(p for p in yt.iterdir() if p.is_dir()):
        if d.name not in PHASES:
            rep.warn(f"youtube-comments/{d.name}/ is not a phase folder ({', '.join(PHASES)}); it is ignored")

    phases_with_data, comment_ids, parents = [], set(), []
    for phase in PHASES:
        files = sorted((yt / phase).glob("*.csv"))
        if not files:
            rep.warn(f"youtube-comments/{phase}/ has no CSVs; the phase is skipped")
            continue
        n_rows = 0
        for f in files:
            loaded = read_csv(f, rep)
            if loaded is None:
                continue
            cols, rows = loaded
            name = f.relative_to(DATA_ROOT)
            missing = [c for c in YOUTUBE_REQUIRED if c not in cols]
            if missing:
                rep.error(f"{name}: missing columns {missing}")
                continue
            bad_time = sum(1 for r in rows if not is_iso_datetime(r["create_time"] or ""))
            if bad_time:
                rep.error(f"{name}: {bad_time} rows with a create_time that is not ISO 8601")
            empty = sum(1 for r in rows if not (r["text"] or "").strip())
            if empty:
                rep.warn(f"{name}: {empty} rows with empty text (dropped in preprocessing)")
            bad_likes = sum(1 for r in rows if r["like_count"] and not r["like_count"].strip().isdigit())
            if bad_likes:
                rep.warn(f"{name}: {bad_likes} rows where like_count is not a whole number")
            for r in rows:
                if r["comment_id"] in comment_ids:
                    rep.error(f"{name}: duplicate comment_id {r['comment_id']}")
                comment_ids.add(r["comment_id"])
                parents.append(r["parent_comment_id"])
            n_rows += len(rows)
        if n_rows:
            phases_with_data.append(phase)
            rep.ok(f"{phase}: {len(files)} videos, {n_rows:,} comments")

    orphans = sum(1 for p in parents if p and p != "root" and p not in comment_ids)
    if orphans:
        rep.warn(f"{orphans} replies point to a parent_comment_id that is not in the data "
                 "(reply context will be empty for those)")
    if not phases_with_data:
        rep.error("no YouTube comments found in any phase folder")
    return phases_with_data


def check_reddit(crisis_dir: Path, rep: Report) -> None:
    print("\nReddit threads (optional)")
    rd = crisis_dir / "reddit-data"
    if not rd.is_dir():
        rep.ok("reddit-data/ not present; skip `preprocess-reddit`")
        return
    allowed = REDDIT_EXTRA_PHASES + PHASES
    for d in sorted(p for p in rd.iterdir() if p.is_dir()):
        if d.name not in allowed:
            rep.warn(f"reddit-data/{d.name}/ is not a phase folder; it is ignored")
    for phase in allowed:
        files = sorted((rd / phase).glob("*/*.csv"))
        if not files:
            continue
        n_threads = 0
        for f in files:
            loaded = read_csv(f, rep)
            if loaded is None:
                continue
            cols, rows = loaded
            name = f.relative_to(DATA_ROOT)
            missing = [c for c in REDDIT_REQUIRED if c not in cols]
            if missing:
                rep.error(f"{name}: missing columns {missing}")
                continue
            posts = sum(1 for r in rows if r["type"] == "post")
            if posts != 1:
                rep.warn(f"{name}: {posts} 'post' rows (expected exactly 1 per thread file)")
            n_threads += 1
        rep.ok(f"{phase}: {n_threads} threads")


def check_transcripts(crisis_dir: Path, rep: Report) -> list[str]:
    print("\nVideo transcripts (optional)")
    vt = crisis_dir / "video-transcripts"
    if not vt.is_dir():
        rep.ok("video-transcripts/ not present; skip the transcript stages")
        return []
    names = []
    for d in sorted(p for p in vt.iterdir() if p.is_dir()):
        if not d.name.startswith("transcripts-"):
            rep.warn(f"video-transcripts/{d.name}/ should be named transcripts-<name>/; it is ignored")
            continue
        vtt = d / "timestamped.vtt"
        if not vtt.is_file():
            rep.error(f"video-transcripts/{d.name}/ has no timestamped.vtt")
            continue
        text = vtt.read_text(encoding="utf-8", errors="replace")
        cues = len(VTT_CUE.findall(text))
        if not text.lstrip().startswith("WEBVTT") or not cues:
            rep.error(f"{vtt.relative_to(DATA_ROOT)}: not a WebVTT file with HH:MM:SS.mmm cues")
            continue
        names.append(d.name.removeprefix("transcripts-"))
        rep.ok(f"{d.name}: {cues} cues")
    return names


def check_pipeline_config(slug: str, phases: list[str], transcripts: list[str], rep: Report) -> None:
    print("\nPipeline config")
    ctx_path = REPO_ROOT / "03-llm-passes/prompts/phase-context.json"
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8")).get(slug, {})
    except (OSError, json.JSONDecodeError):
        ctx = {}
    missing = [p for p in phases if not ctx.get(p)]
    if missing:
        rep.warn(f"phase-context.json has no CASE CONTEXT for {missing}; "
                 "stage 03 (LLM coding) will refuse those phases")
    else:
        rep.ok("phase-context.json covers every phase with data")

    for script in ("01-preprocessing/run_transcripts.sh",
                   "02-analysis/transcripts/run_sentiment.sh",
                   "02-analysis/transcripts/run_topics.sh"):
        body = (REPO_ROOT / script).read_text(encoding="utf-8")
        listed = re.search(r"TRANSCRIPTS=\(\s*(.*?)\)", body, re.S)
        listed_names = listed.group(1).split() if listed else []
        not_listed = [t for t in transcripts if t not in listed_names]
        if not_listed:
            rep.warn(f"{script}: add {not_listed} to its TRANSCRIPTS list")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slug", help="crisis folder name under 00-data/, e.g. ateneo-drowning")
    ap.add_argument("--quiet", action="store_true", help="only print problems and the summary")
    args = ap.parse_args()

    rep = Report(args.quiet)
    crisis_dir = DATA_ROOT / args.slug
    print(f"Checking 00-data/{args.slug}/")
    if not SLUG.match(args.slug):
        rep.warn("slug should be lowercase kebab-case (e.g. flood-response)")
    if not crisis_dir.is_dir():
        rep.error(f"00-data/{args.slug}/ does not exist")
        print(f"\n{rep.errors} error(s), {rep.warnings} warning(s)")
        return 1

    phases = check_youtube(crisis_dir, rep)
    check_reddit(crisis_dir, rep)
    transcripts = check_transcripts(crisis_dir, rep)
    check_pipeline_config(args.slug, phases, transcripts, rep)

    print(f"\n{rep.errors} error(s), {rep.warnings} warning(s)")
    if not rep.errors:
        print(f"Format OK. Next: CRISIS={args.slug} python3 main.py --stage preprocess")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
