"""Crisis Management Analysis Pipeline CLI.

Interactive pipeline runner. Run without arguments for interactive mode,
or use --stage for direct execution.

Examples:
    python main.py                    # interactive prompts
    python main.py --stage pipeline   # full pipeline
    python main.py --stage preprocess # preprocessing only
    python main.py --clean            # wipe outputs
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


# ── Stage definitions ────────────────────────────────────────────────────────

STAGES = {
    # Preprocessing
    "preprocess-youtube":    "01-preprocessing/run_youtube.sh",
    "preprocess-reddit":     "01-preprocessing/run_reddit.sh",
    "preprocess-transcripts": "01-preprocessing/run_transcripts.sh",
    # YouTube analysis
    "sentiment-youtube":     "02-analysis/run_youtube_sentiment.sh",
    "topics-youtube":        "02-analysis/run_youtube_topics.sh",
    # Transcript analysis
    "sentiment-transcripts": "02-analysis/run_transcript_sentiment.sh",
    "topics-transcripts":    "02-analysis/run_transcript_topics.sh",
    # Insights
    "sentiment-change":      "02-analysis/run_sentiment_change.sh",
    "summary":               "02-analysis/run_summary.sh",
}

COMPOSITE_STAGES = {
    "preprocess": ["preprocess-youtube", "preprocess-reddit", "preprocess-transcripts"],
    "analysis":   ["sentiment-youtube", "topics-youtube", "sentiment-transcripts", "topics-transcripts"],
    "insights":   ["sentiment-change", "summary"],
    "pipeline":   ["preprocess", "analysis", "insights"],
}


# ── Execution ────────────────────────────────────────────────────────────────

def run_stage(name: str, force: bool = False) -> bool:
    """Execute a single stage. Returns True on success."""
    if name in COMPOSITE_STAGES:
        for sub in COMPOSITE_STAGES[name]:
            if not run_stage(sub, force):
                return False
        return True

    script = STAGES.get(name)
    if not script:
        print(f"  Unknown stage: {name}")
        return False

    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        print(f"  Script not found: {script}")
        return False

    print(f"\n{'='*50}")
    print(f"  Running: {name}")
    print(f"  Script:  {script}")
    print(f"{'='*50}\n")

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        return False

    print(f"\n  DONE: {name}")
    return True


def clean_outputs():
    """Remove all output files."""
    dirs = [
        PROJECT_ROOT / "01-preprocessing/output/youtube",
        PROJECT_ROOT / "01-preprocessing/output/reddit",
        PROJECT_ROOT / "01-preprocessing/output/transcripts",
        PROJECT_ROOT / "02-analysis/output",
        PROJECT_ROOT / "outputs/figures",
    ]
    patterns = ["*.csv", "*.png"]

    count = 0
    for d in dirs:
        if not d.exists():
            continue
        for pattern in patterns:
            for f in d.glob(pattern):
                f.unlink()
                count += 1

    print(f"\n  Removed {count} output files.")


# ── Interactive mode ─────────────────────────────────────────────────────────

MENU = """
=== Crisis Management Analysis Pipeline ===

  1. Full pipeline (preprocessing → analysis → insights)
  2. Preprocessing
     a. YouTube comments
     b. Reddit posts
     c. Video transcripts
     d. All preprocessing
  3. Analysis
     a. YouTube sentiment
     b. YouTube topics
     c. Transcript sentiment
     d. Transcript topics
     e. All YouTube analysis
     f. All transcript analysis
     g. All analysis
  4. Insights
     a. Sentiment change across phases
     b. Summary
     c. All insights
  5. Clean outputs
  0. Exit
"""

INTERACTIVE_MAP = {
    "1":  "pipeline",
    "2a": "preprocess-youtube",
    "2b": "preprocess-reddit",
    "2c": "preprocess-transcripts",
    "2d": "preprocess",
    "3a": "sentiment-youtube",
    "3b": "topics-youtube",
    "3c": "sentiment-transcripts",
    "3d": "topics-transcripts",
    "3e": ["sentiment-youtube", "topics-youtube"],
    "3f": ["sentiment-transcripts", "topics-transcripts"],
    "3g": "analysis",
    "4a": "sentiment-change",
    "4b": "summary",
    "4c": "insights",
}


def interactive_mode():
    """Run the interactive prompt loop."""
    print(MENU)

    while True:
        try:
            choice = input("Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if choice == "0" or choice == "":
            print("Exiting.")
            break

        if choice == "5":
            clean_outputs()
            print(MENU)
            continue

        stages = INTERACTIVE_MAP.get(choice)
        if stages is None:
            print(f"  Invalid choice: {choice}")
            continue

        if isinstance(stages, str):
            stages = [stages]

        for stage in stages:
            success = run_stage(stage)
            if not success:
                print(f"\nPipeline stopped at: {stage}")
                break

        print(MENU)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crisis Management Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + list(COMPOSITE_STAGES.keys()),
        help="Stage to run (skips interactive prompt)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all output files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs exist",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available stages and exit",
    )
    return parser.parse_args()


def list_stages():
    """Print all available stages."""
    print("\nAvailable stages:")
    print("\n  Composite:")
    for name, subs in COMPOSITE_STAGES.items():
        print(f"    {name:25s} → {' + '.join(subs)}")
    print("\n  Individual:")
    for name, script in STAGES.items():
        print(f"    {name:25s} → {script}")


def main():
    args = parse_args()

    if args.list:
        list_stages()
        return

    if args.clean:
        clean_outputs()
        if not args.stage:
            return

    if args.stage:
        success = run_stage(args.stage, force=args.force)
        sys.exit(0 if success else 1)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
