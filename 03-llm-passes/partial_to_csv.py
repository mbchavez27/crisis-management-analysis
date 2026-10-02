#!/usr/bin/env python3
"""
Converts a partial JSONL file (containing raw LLM chunk responses) into a tabular CSV,
saving the output CSV in the exact same folder as the input file.

Usage:
    python partial_to_csv.py output/ateneo-drowning/full-codebook-coding/01-onset/full_codebook_results.jsonl.partial.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_DEC = json.JSONDecoder()

def load_objs(raw: str) -> list[dict]:
    """Extract every top-level JSON object from the raw LLM output."""
    objs = []
    i = 0
    while (i := raw.find("{", i)) != -1:
        try:
            obj, i = _DEC.raw_decode(raw, i)
            if isinstance(obj, dict):
                objs.append(obj)
        except json.JSONDecodeError:
            i += 1
            continue
    return objs

def main():
    parser = argparse.ArgumentParser(description="Convert partial raw JSONL to CSV in the same folder.")
    parser.add_argument("input_jsonl", type=Path, help="Path to the .partial.jsonl file")
    args = parser.parse_args()

    input_path = args.input_jsonl

    if not input_path.exists():
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)

    # Automatically set the output CSV path in the exact same folder
    # For 'file.partial.jsonl', this creates 'file.partial_extracted.csv'
    output_csv = input_path.with_name(input_path.stem + "_extracted.csv")

    parsed_records = []
    failed_chunks = 0

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                chunk_data = json.loads(line)
                raw_text = chunk_data.get("text", "")
                
                # Strip hallucinated instruction loops if present
                loop_marker = "Now output exactly one JSON object per comment above"
                if loop_marker in raw_text:
                    raw_text = raw_text.split(loop_marker)[0]

                if raw_text:
                    objs = load_objs(raw_text)
                    # Attach the requested row_id if sizes match (best effort mapping)
                    row_ids = chunk_data.get("row_ids", [])
                    if len(objs) == len(row_ids):
                        for rid, obj in zip(row_ids, objs):
                            obj["_row_id"] = rid
                            
                    parsed_records.extend(objs)
            except Exception as e:
                failed_chunks += 1
                continue

    if not parsed_records:
        print("No valid JSON records found in the partial file.")
        sys.exit(0)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(parsed_records)

    # Convert list/dict columns to JSON strings so they format nicely in CSV
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            )

    # Reorder columns to put identifiers first, if they exist
    cols = df.columns.tolist()
    first_cols = [c for c in ["_row_id", "id"] if c in cols]
    other_cols = [c for c in cols if c not in first_cols]
    df = df[first_cols + other_cols]

    # Export
    df.to_csv(output_csv, index=False, encoding="utf-8")
    
    print(f"Success! Extracted {len(parsed_records)} rows.")
    if failed_chunks > 0:
        print(f"Warning: Failed to parse {failed_chunks} chunks.")
    print(f"Saved tabular data to: {output_csv}")

if __name__ == "__main__":
    main()
