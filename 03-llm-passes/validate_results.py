#!/usr/bin/env python3
"""Validate full_codebook_results.csv files from run_local.py against codebook v2.1.

Checks (ERROR = objectively wrong, WARN = likely wrong, INFO = worth a look):
  schema/enums/binaries, template leakage, evidence fidelity, NA consistency,
  emoji-only handling, language heuristics, appearance/anger-emoji rules,
  unused Section H, plus dataset-level patterns (constant columns, repeated
  coding bundles, uniform confidence).

Usage:
    python validate_results.py                      # every full_codebook_results.csv under ./output
    python validate_results.py path/to/results.csv  # one file
    python validate_results.py output/ateneo-drowning --strict   # also exit 1 on WARN

Writes <results>_validation_flags.csv next to each input (one row per flag).
These checks are heuristics: they find rows to review, not a replacement for
comparing against human codes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

BINARY = [
    "ANGER", "CONTEMPT", "DISGUST", "SADNESS", "CONDOLENCE", "COMPASSION_VICTIMS",
    "SYMPATHY_OFFENDER", "FEAR", "NORM_VIOLATION", "IGNORED_WARNING", "RECURRENCE",
    "COVER_UP", "ALT_ACCOUNT", "INJUSTICE", "SELF_INTEREST", "NEEDS_CONTEXT",
    "CLASS_FRAME", "APPEARANCE_ATTACK", "META_DISCOURSE", "SARCASM", "UNCODABLE",
]
SCALARS = {
    "LANGUAGE": {"ENGLISH", "FILIPINO", "TAGLISH", "OTHER_PH_LANGUAGE", "EMOJI", "OTHER"},
    "ATTRIBUTION": {"NONE", "ACCIDENT", "ACCUSATION", "NEGLIGENCE", "INTENTIONAL", "VICTIM_FAULT"},
    "STANCE": {"CRITICAL", "DEFENDING", "MIXED", "NEUTRAL"},
    "CLASS_DIRECTION": {"INEQUALITY_JUSTICE", "VICTIMS_POOR", "INSTITUTION_ELITE", "OTHER", "NA"},
    "RESPONSE_TARGET": {"UNIVERSITY_ANNOUNCEMENT", "COACH_APOLOGY", "PRESIDENT_STATEMENT",
                        "POLICE_REPORT", "OTHER", "NA"},
    "RESPONSE_EVAL": {"ACKNOWLEDGES", "REJECTS", "MIXED", "NA"},
    "TARGET_OF_COMMENT": {"ORGANIZATION", "PERSON", "VICTIMS_FAMILY", "OTHER_COMMENTERS", "MEDIA", "NONE"},
    "confidence": {"HIGH", "MEDIUM", "LOW"},
}
LISTS = {
    "BLAME_TARGETS": {"COACH", "COACHING_STAFF", "UNIVERSITY", "PRESIDENT", "ORGANIZERS", "VENUE",
                      "AUTHORITIES", "VICTIMS", "OTHER", "NONE"},
    "VALUES": {"FAMILY", "CHILD_POTENTIAL", "PARENT_SACRIFICE", "DUTY_OF_CARE", "EDUCATION_MOBILITY",
               "FAITH", "JUSTICE", "SOLIDARITY", "SANCTITY_OF_LIFE", "OTHER", "NONE"},
    "IDENTITY_FRAME": {"NATIONALITY", "RACE", "REGION", "RELIGION", "NONE"},
    "ACTIONS": {"PUNITIVE", "RETALIATION", "RESTITUTION", "ACCOUNTABILITY", "INFORMATION", "REFORM",
                "BOYCOTT", "SUPPORT", "NONE"},
    "REJECT_REASONS": {"INSINCERE", "WORDS_NOT_ENOUGH", "TOO_SOON", "TOO_LATE", "NO_INFORMATION",
                       "SELF_CENTERED", "SHIFTS_BLAME", "DISMISSIVE", "NO_FAMILY_OUTREACH", "OTHER", "NA"},
}
TEXT_COLS = ["row_id", "comment_id", "text", "id", "evidence", "evidence_en", "EMERGENT", "OTHER_NOTE"]
REQUIRED = TEXT_COLS + BINARY + list(SCALARS) + list(LISTS)
NEUTRAL = {"NONE", "NA"}

LEAK = [
    "short quote(s)", "english translation of the evidence", "now output exactly one json",
    "high|medium|low", "supporting each non-zero code", "output only json",
]
FIL = set("""ang ng sa na mga si ay ako ikaw siya mo ka ko namin natin nila nyo niyo po opo hindi wala may
para pero kasi dapat kayo sila lang naman din rin basta yung yun ito iyon pa ba nga sana talaga tayo kami
bakit ano paano tama mali""".split())
EN = set("""the is are was were and to of you your for that this with not but he she they we it in on be have
has should why like will""".split())
SHORTCODE = re.compile(r":[a-z0-9_+\-]+:", re.I)
UNI_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
BALD = re.compile(r"\b(kalbo|kalbohydrate\w*|opaw|upaw|bald|hairless)\b", re.I)
RESPONSE_WORDS = re.compile(
    r"apolog|sorry|sori|patawad|scripted|acting|crocodile|\bwake\b|belated|statement|\bpr\b|lawyer", re.I)
PUNITIVE_WORDS = re.compile(
    r"jail|prison|kulong|kasuhan|\bkaso\b|managot|punish|parusahan|charge|arrest|deport|resign|"
    r"\bfire[dm]?\b|\bban(ned)?\b|revoke|accountab|surrender|\bsue\b|panagutin|sagutin", re.I)
ANGER_EMOJI = {"enraged_face", "angry_face", "face_with_symbols_on_mouth", "pouting_face"}
SAD_EMOJI = {"crying_face", "loudly_crying_face", "broken_heart", "pensive_face", "disappointed_face"}


def parse_list(val: str):
    try:
        v = json.loads(val)
    except Exception:
        return None
    return v if isinstance(v, list) and all(isinstance(x, str) for x in v) else None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" \"'\u201c\u201d")


def find_files(args_paths: list[str]) -> list[Path]:
    roots = [Path(p) for p in args_paths] or [HERE / "output"]
    files: list[Path] = []
    for p in roots:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("full_codebook_results.csv")))
    return files


def validate(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    flags: list[dict] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        flags.append({"row_id": "", "comment_id": "", "severity": "ERROR",
                      "check": "missing_columns", "detail": str(missing)})
        return df, flags
    dups = set(df.loc[df["comment_id"].duplicated(keep=False), "comment_id"])
    sigs: list[tuple] = []

    for _, s in df.iterrows():
        r = s.to_dict()

        def F(sev: str, check: str, detail: str = "") -> None:
            flags.append({"row_id": r["row_id"], "comment_id": r["comment_id"],
                          "severity": sev, "check": check, "detail": detail})

        if r["id"] == "":
            F("ERROR", "uncoded_row", "no model output for this row")
            continue
        if r["id"] != r["comment_id"]:
            F("ERROR", "id_mismatch", f"{r['id']!r} vs {r['comment_id']!r}")
        if r["comment_id"] in dups:
            F("ERROR", "duplicate_comment_id")

        for c in BINARY:
            if r[c] not in ("0", "1"):
                F("ERROR", "bad_binary", f"{c}={r[c]!r}")
        for c, allowed in SCALARS.items():
            if r[c] not in allowed:
                F("ERROR", "bad_enum", f"{c}={r[c]!r}")
        L: dict[str, list[str]] = {}
        for c, allowed in LISTS.items():
            v = parse_list(r[c])
            if v is None or not v:
                F("ERROR", "bad_list", f"{c}={r[c]!r}")
                L[c] = []
                continue
            bad = [x for x in v if x not in allowed]
            if bad:
                F("ERROR", "bad_enum", f"{c} has {bad}")
            if len(v) > 1 and NEUTRAL & set(v):
                F("WARN", "none_mixed_with_values", f"{c}={v}")
            L[c] = v

        def b(c: str) -> bool:
            return r[c] == "1"

        # --- template / prompt leakage -------------------------------------
        leaked_ev = False
        for c in ("evidence", "evidence_en", "EMERGENT", "OTHER_NOTE", "confidence"):
            low = r[c].lower()
            if any(p in low for p in LEAK):
                F("ERROR", "template_leakage", f"{c}: {r[c][:70]!r}")
                leaked_ev = leaked_ev or c == "evidence"

        # --- evidence fidelity ---------------------------------------------
        ev = r["evidence"].strip()
        coded_any = (any(b(c) for c in BINARY if c != "UNCODABLE") or r["ATTRIBUTION"] != "NONE"
                     or L.get("ACTIONS") != ["NONE"] or L.get("VALUES") != ["NONE"])
        if ev and not leaked_ev:
            t = norm(r["text"])
            frags = [norm(f) for f in re.split(r"\s+\|\s+|;\s+|\s+/\s+", ev) if f.strip()]
            miss = [f for f in frags if f not in t]
            if miss:
                F("WARN", "evidence_not_in_text", f"{miss[0][:70]!r}")
        if coded_any and not ev:
            F("WARN", "no_evidence", "codes set but evidence is empty")
        if ev and not r["evidence_en"].strip():
            F("WARN", "no_evidence_en")
        if "@" in ev:
            F("WARN", "handle_in_evidence", "replace @handles with [USER]")

        # --- structural consistency ----------------------------------------
        if b("UNCODABLE") and (coded_any or r["STANCE"] != "NEUTRAL"):
            F("ERROR", "uncodable_not_blank", "UNCODABLE=1 requires all other codes blank")
        if b("NEEDS_CONTEXT") and r["STANCE"] != "NEUTRAL":
            F("ERROR", "needs_context_stance", f"STANCE={r['STANCE']} (rule 8: must be NEUTRAL)")
        if (r["CLASS_FRAME"] == "0") != (r["CLASS_DIRECTION"] == "NA"):
            F("ERROR", "class_na_mismatch", f"CLASS_FRAME={r['CLASS_FRAME']} CLASS_DIRECTION={r['CLASS_DIRECTION']}")
        if (r["RESPONSE_EVAL"] == "NA") != (r["RESPONSE_TARGET"] == "NA"):
            F("ERROR", "response_na_mismatch", f"TARGET={r['RESPONSE_TARGET']} EVAL={r['RESPONSE_EVAL']}")
        rr = L.get("REJECT_REASONS", [])
        if r["RESPONSE_EVAL"] in ("REJECTS", "MIXED") and rr == ["NA"]:
            F("WARN", "rejects_without_reason", "REJECT_REASONS is NA")
        if r["RESPONSE_EVAL"] not in ("REJECTS", "MIXED") and rr and rr != ["NA"]:
            F("ERROR", "reject_reasons_without_rejection", str(rr))
        if r["ATTRIBUTION"] in ("ACCUSATION", "NEGLIGENCE", "INTENTIONAL") and not b("NORM_VIOLATION"):
            F("WARN", "attribution_without_norm_violation", r["ATTRIBUTION"])

        # --- content heuristics --------------------------------------------
        text = r["text"]
        codes = {x.lower() for x in SHORTCODE.findall(text)}
        codes = {x.strip(":") for x in codes}
        bare = UNI_EMOJI.sub("", SHORTCODE.sub("", text))
        bare = re.sub(r"[\W_]+", "", bare)
        if text.strip() and not bare:
            if r["LANGUAGE"] != "EMOJI":
                F("ERROR", "emoji_only_language", f"LANGUAGE={r['LANGUAGE']}, expected EMOJI")
            extra = [k for k, v in (("NORM_VIOLATION", b("NORM_VIOLATION")),
                                    ("ATTRIBUTION", r["ATTRIBUTION"] != "NONE"),
                                    ("BLAME_TARGETS", L.get("BLAME_TARGETS") != ["NONE"]),
                                    ("VALUES", L.get("VALUES") != ["NONE"]),
                                    ("ACTIONS", L.get("ACTIONS") != ["NONE"])) if v]
            if extra:
                F("WARN", "emoji_only_overcoding", f"emoji-only comment coded {extra}")
        else:
            tokens = re.findall(r"[a-z\u00f1']+", SHORTCODE.sub(" ", text.lower()))
            fil_n = sum(t in FIL for t in tokens)
            en_n = sum(t in EN for t in tokens)
            if r["LANGUAGE"] == "FILIPINO" and fil_n == 0 and en_n >= 2:
                F("WARN", "language_heuristic", "coded FILIPINO but reads English/mixed (ENGLISH or TAGLISH?)")
            elif r["LANGUAGE"] == "ENGLISH" and fil_n >= 2:
                F("WARN", "language_heuristic", "coded ENGLISH but has Filipino words (TAGLISH?)")
            elif r["LANGUAGE"] == "TAGLISH" and (fil_n == 0 or en_n == 0) and fil_n + en_n >= 2:
                F("WARN", "language_heuristic", "coded TAGLISH but only one language detected")
        if BALD.search(text) and not b("APPEARANCE_ATTACK"):
            F("WARN", "appearance_missed", f"matched {BALD.search(text).group(0)!r}")
        if codes & ANGER_EMOJI and not b("ANGER"):
            F("WARN", "anger_emoji_missed", str(sorted(codes & ANGER_EMOJI)))
        if codes & SAD_EMOJI and not b("SADNESS"):
            F("INFO", "sad_emoji_missed", str(sorted(codes & SAD_EMOJI)))
        if RESPONSE_WORDS.search(text) and r["RESPONSE_EVAL"] == "NA":
            F("WARN", "possible_response_uncoded", f"matched {RESPONSE_WORDS.search(text).group(0)!r}")
        if "PUNITIVE" in L.get("ACTIONS", []) and not PUNITIVE_WORDS.search(text):
            F("INFO", "punitive_no_keyword", "PUNITIVE coded, no punishment wording found")

        sigs.append((r["NORM_VIOLATION"], r["ATTRIBUTION"], tuple(sorted(L.get("BLAME_TARGETS", []))),
                     r["STANCE"], tuple(sorted(L.get("ACTIONS", [])))))

    # --- dataset-level patterns ---------------------------------------------
    ok = df[df["id"] != ""]
    n = len(ok)

    def D(sev: str, check: str, detail: str) -> None:
        flags.append({"row_id": "", "comment_id": "", "severity": sev, "check": check, "detail": detail})

    if n >= 10:
        for c in ["TARGET_OF_COMMENT", "RESPONSE_EVAL", "SARCASM", "META_DISCOURSE", "CLASS_FRAME", "confidence"]:
            if ok[c].nunique() == 1:
                D("WARN", "constant_column", f"{c} is {ok[c].iloc[0]!r} on all {n} rows (legitimate?)")
        top, cnt = Counter(sigs).most_common(1)[0]
        if cnt / n > 0.4:
            D("WARN", "repeated_bundle", f"{cnt}/{n} rows share NORM_VIOLATION/ATTRIBUTION/BLAME/STANCE/ACTIONS = {top}")
        if (ok["confidence"] == "HIGH").mean() > 0.95:
            D("WARN", "uniform_confidence", "over 95% HIGH; confidence cannot be used for filtering")
    return df, flags


def report(path: Path, df: pd.DataFrame, flags: list[dict]) -> int:
    ok = df[df["id"] != ""] if "id" in df.columns else df
    n_all, n = len(df), len(ok)
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    bad_rows = {f["row_id"] for f in flags if f["severity"] in ("ERROR", "WARN") and f["row_id"] != ""}
    print(f"rows: {n_all} | coded: {n} | rows with ERROR/WARN: {len(bad_rows)} | clean: {n_all - len(bad_rows)}")

    by = Counter((f["severity"], f["check"]) for f in flags)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    print("\nFlags:")
    if not by:
        print("  none")
    for (sev, chk), cnt in sorted(by.items(), key=lambda kv: (order[kv[0][0]], -kv[1])):
        ex = [f for f in flags if f["severity"] == sev and f["check"] == chk][:3]
        rows = ",".join(str(f["row_id"]) for f in ex if f["row_id"] != "")
        det = ex[0]["detail"][:80] if ex else ""
        print(f"  [{sev:5}] {chk:34} x{cnt:<4} rows: {rows or '-':14} {det}")

    if n and all(c in ok.columns for c in BINARY):
        print("\nBinary code rates (coded rows):")
        line = [f"{c}={(ok[c] == '1').sum()}({(ok[c] == '1').mean():.0%})" for c in BINARY]
        for i in range(0, len(line), 4):
            print("  " + "  ".join(line[i:i + 4]))
        print("\nCategorical distributions:")
        for c in SCALARS:
            if c in ok.columns:
                print(f"  {c}: { {k: int(v) for k, v in ok[c].value_counts().items()} }")
        for c in LISTS:
            if c in ok.columns:
                cnt = Counter(x for v in ok[c] for x in (parse_list(v) or []))
                print(f"  {c}: {dict(cnt.most_common())}")

    out = path.with_name(path.stem + "_validation_flags.csv")
    pd.DataFrame(flags, columns=["row_id", "comment_id", "severity", "check", "detail"]).to_csv(out, index=False)
    print(f"\nFlags written -> {out}")
    return sum(1 for f in flags if f["severity"] == "ERROR"), sum(1 for f in flags if f["severity"] == "WARN")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="results CSV files or directories (default: ./output)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as ERROR")
    args = ap.parse_args()
    files = find_files(args.paths)
    if not files:
        print("No full_codebook_results.csv files found.")
        return 2
    errs = warns = 0
    for f in files:
        df, flags = validate(f)
        e, w = report(f, df, flags)
        errs += e
        warns += w
    print(f"\nOverall: {errs} ERROR flags, {warns} WARN flags across {len(files)} file(s)")
    return 1 if errs or (args.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
