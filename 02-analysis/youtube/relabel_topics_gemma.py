"""Relabel YouTube topic clusters with Gemma via the Gemini API.

Re-reads the per-phase topics CSVs, labels every non-noise topic with
``gemma-4-31b-it`` using the same prompt as the topic-modeling notebook,
and writes the CSVs back in place. Checkpointed per phase so a rerun
skips already-labeled phases (API quota safety).

Usage:
    uv run --project 02-analysis python 02-analysis/youtube/relabel_topics_gemma.py
    CRISIS=other-slug uv run --project 02-analysis python 02-analysis/youtube/relabel_topics_gemma.py
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

NOTEBOOK_DIR = Path(__file__).resolve().parent
CRISIS = os.environ.get("CRISIS", "ateneo-drowning")
OUTPUT_DIR = NOTEBOOK_DIR / ".." / "output" / CRISIS / "youtube-comments" / "topic-modeling"
CHECKPOINT = OUTPUT_DIR / ".relabel-gemma-checkpoint.json"
MODEL = "gemma-4-26b-a4b-it"  # 31b persistent 500s on 2026-10-13; 26b healthy
PHASES = ["01-onset", "02-confirmation", "03-spread", "04-reckoning", "05-lingering"]


def label_topic(client: genai.Client, topic_id: int, term_str: str, samples: list[str]) -> str:
    """Request a 2-4 word label with retries on transient API errors."""
    prompt = (
        "You are a crisis discourse analyst. Given the top terms "
        "and sample comments for a topic cluster, assign a short "
        "label (2-4 words).\n\n"
        f"Top terms: {term_str}\n"
        "Sample comments:\n"
        + "\n".join(f"- {s[:200]}" for s in samples[:5])
        + "\n\nLabel (2-4 words):"
    )
    for attempt in (1, 2, 3, 4, 5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            break
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("429", "503", "500", "overloaded", "overload", "INTERNAL"))
            if transient and attempt < 5:
                wait = 60 * 2 ** (attempt - 1)
                print(f"  transient error, sleeping {wait}s (attempt {attempt}/5)...", flush=True)
                time.sleep(wait)
            else:
                raise
    return response.text.strip().strip("*").strip()


def main() -> None:
    """Relabel all phases, skipping checkpointed ones."""
    load_dotenv(NOTEBOOK_DIR / ".." / ".." / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    done: dict[str, dict[str, str]] = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}

    for phase in PHASES:
        if phase in done:
            print(f"{phase}: already labeled, skipping")
            continue
        path = OUTPUT_DIR / f"topics_{phase}.csv"
        df = pd.read_csv(path)
        labels: dict[str, str] = {}
        for tid in sorted(t for t in set(df["topic_id"]) if t != -1):
            mask = df["topic_id"] == tid
            term_str = str(df.loc[mask, "top_terms"].iloc[0])
            samples = df.loc[mask, "text"].sample(min(5, int(mask.sum())), random_state=42).tolist()
            label = label_topic(client, int(tid), term_str, samples)
            labels[str(int(tid))] = label
            df.loc[mask, "topic_label"] = label
            print(f"{phase} topic {tid}: {label}", flush=True)
            time.sleep(13)  # free tier pacing
        df.to_csv(path, index=False)
        done[phase] = labels
        CHECKPOINT.write_text(json.dumps(done, indent=2))
        print(f"{phase}: saved")

    print("All phases relabeled.")


if __name__ == "__main__":
    main()
