#!/usr/bin/env python3
"""Validate run-class-emotions.py results against codebook v2.1 and describe the dataset.

Checks (ERROR = objectively wrong, WARN = likely wrong, INFO = worth a look):
  schema/enums/binaries, template leakage, evidence fidelity (old free-text and
  v1.3+ «quote» -> CODES formats), NA consistency, emoji-only handling, language
  heuristics, appearance/anger-emoji rules, unused Section H, plus dataset-level
  patterns (constant columns, repeated coding bundles, uniform confidence, mixed
  prompt versions or models).

Descriptive statistics, per phase and for all selected phases combined:
  coverage and failures, comment length, videos, language, code prevalence,
  categorical and multi-label distributions, confidence, review flags.

Usage (run from 03-llm-passes/):
    python scripts/class-emotions/validate-class-emotions.py                                  # all phases
    python scripts/class-emotions/validate-class-emotions.py --phases 01-onset                # one phase
    python scripts/class-emotions/validate-class-emotions.py --phases 01-onset,04-reckoning   # several
    python scripts/class-emotions/validate-class-emotions.py --phases all --strict            # exit 1 on WARN too
    python scripts/class-emotions/validate-class-emotions.py path/to/full_codebook_results.csv   # explicit file(s)

Outputs:
  <phase>/full_codebook_results_validation_flags.csv   one row per flag (per phase)
  <pass dir>/_validation/<scope>/flags_all.csv          all flags, with a phase column
  <pass dir>/_validation/<scope>/phase_overview.csv     one row per phase (+ ALL if several)
  <pass dir>/_validation/<scope>/stats_long.csv         phase x variable x category counts
  (<scope> is "all", or the selected phases joined with "+")

These checks are heuristics: they find rows to review. Accuracy still has to be
measured against human codes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_NAME = "full_codebook_results.csv"

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
# Columns shown side by side in the cross-phase table.
KEY_RATES = ["ANGER", "CONTEMPT", "SADNESS", "CONDOLENCE", "NORM_VIOLATION", "INJUSTICE", "SARCASM"]

LEAK = [
    "short quote(s)", "english translation of the evidence", "now output exactly one json",
    "high|medium|low", "supporting each non-zero code", "output only json",
    "«exact words»", "«english translation»", "exact words from the comment",
    "literal english translation",
]
FIL = set("""ang ng sa na mga si ay ako ikaw siya mo ka ko namin natin nila nyo niyo po opo hindi wala may
para pero kasi dapat kayo sila lang naman din rin basta yung yun ito iyon pa ba nga sana talaga tayo kami
bakit ano paano tama mali""".split())
EN = set("""the is are was were and to of you your for that this with not but he she they we it in on be have
has should why like will""".split())
SHORTCODE = re.compile(r":[a-z0-9_+\-]+:", re.I)
UNI_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
EVID_QUOTE = re.compile(r"«([^«»]*)»")
BALD = re.compile(r"\b(kalbo|kalbohydrate\w*|opaw|upaw|bald|hairless)\b", re.I)
RESPONSE_WORDS = re.compile(
    r"apolog|sorry|sori|patawad|scripted|acting|crocodile|\bwake\b|belated|statement|\bpr\b|lawyer", re.I)
PUNITIVE_WORDS = re.compile(
    r"jail|prison|kulong|kasuhan|\bkaso\b|managot|punish|parusahan|charge|arrest|deport|resign|"
    r"\bfire[dm]?\b|\bban(ned)?\b|revoke|accountab|surrender|\bsue\b|panagutin|sagutin", re.I)
# Class vocabulary (F1: class, wealth, poverty, privilege, scholarship status, rich vs
# poor). Used only to AUDIT CLASS_FRAME coverage, never to code. Extend as needed.
CLASS_WORDS = re.compile(
    r"\b(?:mahirap|mahihirap|mayaman|mayayaman|elite|elitist|privileged?|privilege|nepo\w*|"
    r"scholar\w*|iskolar|pera|money|millions?|billions?|rich|poor|probinsya\w*|"
    r"working class|middle class|tuition|matrikula|afford\w*|breadwinner|classism|burgis|"
    r"connections?|palakasan|kapit|kahirapan|dukha|maralita|sosyal|coñ?o)\b", re.I)
EMOTIONS = ["ANGER", "CONTEMPT", "DISGUST", "SADNESS", "CONDOLENCE", "COMPASSION_VICTIMS",
            "SYMPATHY_OFFENDER", "FEAR"]
# review_flags written by run-class-emotions.py v1.5 (rows kept after correction retries).
REVIEW_SEVERITY = {"evidence_unverified": "WARN", "class_direction_missing": "WARN",
                   "response_incomplete": "WARN", "class_direction_without_frame": "WARN",
                   "evidence_long": "INFO", "evidence_fuzzy": "INFO", "other_unlabeled": "INFO",
                   "blame_unnamed": "WARN", "class_frame_no_class_word": "INFO"}
NEEDS_CONTEXT_MAX = 0.15
LEN_BINS = [(0, 20, "words_le20"), (21, 80, "words_21_80"), (81, 10**9, "words_gt80")]
ANGER_EMOJI = {"enraged_face", "angry_face", "face_with_symbols_on_mouth", "pouting_face"}
SAD_EMOJI = {"crying_face", "loudly_crying_face", "broken_heart", "pensive_face", "disappointed_face"}
FLAG_COLS = ["phase", "row_id", "comment_id", "severity", "check", "detail"]


# --------------------------------------------------------------------------- helpers
def parse_list(val: str):
    try:
        v = json.loads(val)
    except Exception:
        return None
    return v if isinstance(v, list) and all(isinstance(x, str) for x in v) else None


def norm(s: str) -> str:
    s = re.sub("[\"'`\\\u2018\u2019\u201c\u201d]", "", s.replace("\ufe0f", ""))
    return re.sub(r"\s+", " ", s.lower()).strip(" .,!?;:")


def is_emoji_only(text: str) -> bool:
    bare = re.sub(r"[\W_]+", "", UNI_EMOJI.sub("", SHORTCODE.sub("", text)))
    return bool(text.strip()) and not bare


def evidence_fragments(ev: str) -> list[str]:
    """Quotes from either evidence format: «quote» -> CODES (v1.2+) or free text (v1.0/1.1)."""
    if "«" in ev:
        return [q for q in EVID_QUOTE.findall(ev) if q.strip()]
    return [f for f in re.split(r"\s+\|\s+|;\s+|\s+/\s+", ev) if f.strip()]


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def read_jsonl_field(path: Path, field: str) -> Counter:
    out: Counter = Counter()
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out[str(json.loads(line).get(field, "unrecorded"))] += 1
            except Exception:
                continue
    return out


_REASON = re.compile(r"([a-z][a-z_]{2,})(?::|')")


def failure_reasons(path: Path) -> Counter:
    """Count hard failures by reason (prompt_too_long, evidence_not_verbatim, ...)."""
    out: Counter = Counter()
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                err = str(json.loads(line).get("error", ""))
            except Exception:
                continue
            if "EngineDeadError" in err:
                out["engine_dead"] += 1
                continue
            m = _REASON.search(err.split("row ", 1)[-1] if "row " in err else err)
            out[m.group(1) if m else "other"] += 1
    return out


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# --------------------------------------------------------------------------- discovery
def discover(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """(phase, results.csv) pairs from explicit paths or --crisis/--phases."""
    if args.paths:
        found: list[Path] = []
        for p in map(Path, args.paths):
            if p.is_file():
                found.append(p)
            elif p.is_dir():
                found.extend(sorted(p.rglob(RESULTS_NAME)))
        return [(p.parent.name, p) for p in found]

    pass_dir = args.output_root / args.crisis / args.pass_dir
    if not pass_dir.is_dir():
        print(f"No results directory: {pass_dir}")
        return []
    available = sorted(d.name for d in pass_dir.iterdir()
                       if d.is_dir() and (d / RESULTS_NAME).exists())
    if args.phases.strip().lower() == "all":
        wanted = available
    else:
        wanted = [p.strip() for p in args.phases.split(",") if p.strip()]
        missing = [p for p in wanted if p not in available]
        if missing:
            print(f"WARNING: no results for {missing}. Available: {available}")
        wanted = [p for p in wanted if p in available]
    return [(p, pass_dir / p / RESULTS_NAME) for p in wanted]


# --------------------------------------------------------------------------- validation
def validate(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # run_local.py versions before the fix wrote binaries as 0.0 / 1.0 when a phase
    # had uncoded rows (pandas upcasts to float on the merge). Same values, so read
    # them as 0 / 1 instead of flagging every cell.
    for c in BINARY:
        if c in df.columns:
            df[c] = df[c].replace({"0.0": "0", "1.0": "1"})
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
        review = [x for x in r.get("review_flags", "").split(";") if x]
        for rf in review:
            base = rf.split("(")[0]
            F(REVIEW_SEVERITY.get(base, "INFO"), f"review:{base}", rf)
        ev_flagged = any(x.startswith(("evidence_unverified", "evidence_fuzzy")) for x in review)
        if ev and not leaked_ev and not ev_flagged:
            t = norm(r["text"].replace("\n", " "))
            miss = [f for f in (norm(x) for x in evidence_fragments(ev)) if f and f not in t]
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
            if "class_direction_missing" in review:
                pass  # kept on purpose by run_local (already flagged for review)
            else:
                F("ERROR", "class_na_mismatch",
                  f"CLASS_FRAME={r['CLASS_FRAME']} CLASS_DIRECTION={r['CLASS_DIRECTION']}")
        if r["CLASS_FRAME"] == "0" and CLASS_WORDS.search(r["text"]):
            F("INFO", "class_word_not_framed", f"matched {CLASS_WORDS.search(r['text']).group(0)!r}")
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
        codes = {x.lower().strip(":") for x in SHORTCODE.findall(text)}
        if is_emoji_only(text):
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
        nc = (ok["NEEDS_CONTEXT"] == "1").mean()
        if nc > NEEDS_CONTEXT_MAX:
            D("WARN", "needs_context_high",
              f"NEEDS_CONTEXT on {nc:.0%} of rows (> {NEEDS_CONTEXT_MAX:.0%}); rule 8 forces STANCE=NEUTRAL")
        kw = df["text"].str.contains(CLASS_WORDS)
        if kw.sum() >= 20:
            miss_kw = (df.loc[kw, "id"] == "").mean()
            miss_other = (df.loc[~kw, "id"] == "").mean()
            if miss_kw > 1.5 * miss_other and miss_kw > 0.05:
                D("WARN", "class_comments_missing",
                  f"{miss_kw:.0%} of class-vocabulary comments uncoded vs {miss_other:.0%} of others")
        for c in ["STANCE", "TARGET_OF_COMMENT", "LANGUAGE", "RESPONSE_EVAL", "SARCASM",
                  "META_DISCOURSE", "CLASS_FRAME", "confidence"]:
            if ok[c].nunique() == 1:
                D("WARN", "constant_column", f"{c} is {ok[c].iloc[0]!r} on all {n} rows (legitimate?)")
        if sigs:
            top, cnt = Counter(sigs).most_common(1)[0]
            if cnt / n > 0.4:
                D("WARN", "repeated_bundle",
                  f"{cnt}/{n} rows share NORM_VIOLATION/ATTRIBUTION/BLAME/STANCE/ACTIONS = {top}")
        if (ok["confidence"] == "HIGH").mean() > 0.95:
            D("WARN", "uniform_confidence", "over 95% HIGH; confidence cannot be used for filtering")
    return df, flags


# --------------------------------------------------------------------------- statistics
def describe(phase: str, df: pd.DataFrame, flags: list[dict], results_path: Path | None) -> tuple[dict, list[dict]]:
    """One overview row and long-format distribution rows for a phase (or ALL)."""
    ok = df[df["id"] != ""] if "id" in df.columns else df
    n_all, n = len(df), len(ok)
    words = df["text"].str.split().str.len().fillna(0) if "text" in df.columns else pd.Series(dtype=float)
    chars = df["text"].str.len() if "text" in df.columns else pd.Series(dtype=float)
    flag_rows = {(f.get("phase", phase), f["row_id"]) for f in flags
                 if f["severity"] in ("ERROR", "WARN") and f["row_id"] != ""}

    ov: dict[str, Any] = {
        "phase": phase,
        "rows": n_all,
        "coded": n,
        "coverage_pct": pct(n, n_all),
        "errors": sum(f["severity"] == "ERROR" for f in flags),
        "warnings": sum(f["severity"] == "WARN" for f in flags),
        "rows_flagged": len(flag_rows),
        "rows_flagged_pct": pct(len(flag_rows), n_all),
        "videos": df["video_id"].nunique() if "video_id" in df.columns else None,
        "words_mean": round(float(words.mean()), 1) if len(words) else None,
        "words_median": float(words.median()) if len(words) else None,
        "words_max": int(words.max()) if len(words) else None,
        "chars_mean": round(float(chars.mean()), 1) if len(chars) else None,
        "emoji_only": int(df["text"].map(is_emoji_only).sum()) if "text" in df.columns else None,
    }
    # Sampled runs (run_local.py --sample-per-phase / --class-census) carry per-row
    # weights; the class census oversamples class talk, so unweighted percentages
    # would overstate class-related codes. Percentages are weighted when weights exist.
    weighted = "sample_weight" in ok.columns and ok["sample_weight"].astype(str).str.strip().ne("").any()
    w = (pd.to_numeric(ok["sample_weight"], errors="coerce").fillna(1.0) if weighted
         else pd.Series(1.0, index=ok.index))
    ov["weighted"] = bool(weighted)
    if weighted:
        strata = ok["sample_stratum"].value_counts().to_dict() if "sample_stratum" in ok.columns else {}
        ov["sample_strata"] = ";".join(f"{k}:{v}" for k, v in strata.items())
        ov["population_N_est"] = round(float(w.sum()))

    def wpct(mask: pd.Series) -> float:
        return round(100.0 * float(w[mask].sum()) / float(w.sum()), 1) if len(w) and w.sum() else 0.0

    if "text" in df.columns and "id" in df.columns:
        words_all = df["text"].str.split().str.len().fillna(0)
        for lo, hi, name in LEN_BINS:
            m = words_all.between(lo, hi)
            ov[f"uncoded_pct_{name}"] = pct(int((df.loc[m, "id"] == "").sum()), int(m.sum()))
        kw_all = df["text"].str.contains(CLASS_WORDS)
        ov["class_vocab_comments"] = int(kw_all.sum())
        ov["class_vocab_uncoded_pct"] = pct(int((df.loc[kw_all, "id"] == "").sum()), int(kw_all.sum()))
    if n and "CLASS_FRAME" in ok.columns:
        kw = ok["text"].str.contains(CLASS_WORDS)
        ov["CLASS_FRAME_pct"] = wpct(ok["CLASS_FRAME"] == "1")
        ov["class_vocab_framed_pct"] = pct(int(((ok["CLASS_FRAME"] == "1") & kw).sum()), int(kw.sum()))
        ov["NEEDS_CONTEXT_pct"] = pct(int((ok["NEEDS_CONTEXT"] == "1").sum()), n)
    if "review_flags" in ok.columns:
        rf = ok["review_flags"].fillna("")
        ov["review_flagged_rows"] = int((rf != "").sum())
        kinds = Counter(x.split("(")[0] for v in rf for x in v.split(";") if x)
        ov["review_flag_types"] = ";".join(f"{k}:{v}" for k, v in kinds.most_common())
    if "autofix" in ok.columns:
        fixes = ok["autofix"].fillna("")
        ov["language_review_flags"] = int(fixes.str.contains("language_check").sum())
        ov["autofixed_rows"] = int(fixes.map(
            lambda s: any(f and not f.startswith("language_check") for f in s.split(";"))).sum())
    if results_path is not None:
        d = results_path.parent
        versions = read_jsonl_field(d / "full_codebook_results.jsonl", "prompt_version")
        models = read_jsonl_field(d / "full_codebook_results.jsonl", "model")
        precisions = read_jsonl_field(d / "full_codebook_results.jsonl", "precision")
        ov["precisions"] = ";".join(f"{k}:{v}" for k, v in sorted(precisions.items()))
        ov["prompt_versions"] = ";".join(f"{k}:{v}" for k, v in sorted(versions.items()))
        ov["models"] = ";".join(f"{k}:{v}" for k, v in sorted(models.items()))
        ov["hard_failures"] = count_lines(d / "full_codebook_failures.log")
        reasons = failure_reasons(d / "full_codebook_failures.log")
        ov["failure_reasons"] = ";".join(f"{k}:{v}" for k, v in reasons.most_common())
        ov["pending"] = count_lines(d / "full_codebook_pending.log")

    long: list[dict] = []
    def add(var: str, kind: str, cat: str, k: int, mask: pd.Series | None = None) -> None:
        long.append({"phase": phase, "variable": var, "type": kind, "category": cat, "n": int(k),
                     "pct_of_coded": wpct(mask) if mask is not None else pct(int(k), n),
                     "pct_unweighted": pct(int(k), n), "weighted": bool(weighted)})

    if n:
        for c in BINARY:
            if c in ok.columns:
                m = ok[c] == "1"
                add(c, "binary", "1", m.sum(), m)
        for c in SCALARS:
            if c in ok.columns:
                for cat, k in ok[c].value_counts().items():
                    add(c, "categorical", str(cat), k, ok[c] == cat)
        for c in LISTS:
            if c in ok.columns:
                parsed = ok[c].map(lambda v: parse_list(v) or [])
                for cat, k in Counter(x for v in parsed for x in v).most_common():
                    add(c, "multi_label", cat, k, parsed.map(lambda v, cat=cat: cat in v))
                real = parsed.map(lambda v: [x for x in v if x not in NEUTRAL])
                ov[f"{c}_labels_per_row"] = round(float(real.str.len().mean()), 2)
        if "CLASS_FRAME" in ok.columns:
            for frame, grp in (("class_framed", ok[ok["CLASS_FRAME"] == "1"]),
                               ("not_class_framed", ok[ok["CLASS_FRAME"] != "1"])):
                for e in EMOTIONS + ["NORM_VIOLATION", "INJUSTICE"]:
                    if e in grp.columns and len(grp):
                        k = int((grp[e] == "1").sum())
                        long.append({"phase": phase, "variable": f"{e}|{frame}", "type": "class_x_code",
                                     "category": "1", "n": k, "pct_of_coded": pct(k, len(grp))})
        for c in KEY_RATES:
            if c in ok.columns:
                ov[f"{c}_pct"] = wpct(ok[c] == "1")
        for cat in ("CRITICAL", "DEFENDING", "MIXED", "NEUTRAL"):
            ov[f"STANCE_{cat}_pct"] = wpct(ok["STANCE"] == cat) if "STANCE" in ok.columns else None
    return ov, long


# --------------------------------------------------------------------------- reporting
def print_phase(phase: str, path: Path, ov: dict, long: list[dict], flags: list[dict], show_dist: bool) -> None:
    print(f"\n{'=' * 78}\n{phase}  ({path})\n{'=' * 78}")
    print(f"rows {ov['rows']} | coded {ov['coded']} ({ov['coverage_pct']}%) | "
          f"rows with ERROR/WARN {ov['rows_flagged']} ({ov['rows_flagged_pct']}%) | "
          f"hard failures {ov.get('hard_failures', '-')} | pending {ov.get('pending', '-')}")
    print(f"comments: {ov['videos']} videos | words mean {ov['words_mean']}, median {ov['words_median']}, "
          f"max {ov['words_max']} | emoji-only {ov['emoji_only']}")
    if ov.get("weighted"):
        print(f"SAMPLED: strata {ov.get('sample_strata')} | estimated population {ov.get('population_N_est')} "
              f"| percentages below are WEIGHTED by sample_weight")
    if ov.get("failure_reasons"):
        print(f"hard-failure reasons: {ov['failure_reasons']}")
    if ov.get("prompt_versions"):
        print(f"prompt versions: {ov['prompt_versions']} | models: {ov['models']} | "
              f"precision: {ov.get('precisions') or 'unrecorded'}")
    if "language_review_flags" in ov:
        print(f"LANGUAGE review flags: {ov['language_review_flags']} | autofixed rows: {ov['autofixed_rows']}")
    if "review_flagged_rows" in ov:
        print(f"kept with review flags: {ov['review_flagged_rows']} ({ov.get('review_flag_types') or '-'})")
    if "uncoded_pct_words_le20" in ov:
        print(f"uncoded by length: <=20 words {ov['uncoded_pct_words_le20']}% | 21-80 "
              f"{ov['uncoded_pct_words_21_80']}% | >80 {ov['uncoded_pct_words_gt80']}%")
    if "CLASS_FRAME_pct" in ov:
        print(f"class audit: CLASS_FRAME {ov['CLASS_FRAME_pct']}% of coded | class-vocabulary comments "
              f"{ov['class_vocab_comments']} (uncoded {ov['class_vocab_uncoded_pct']}%, "
              f"class-framed {ov['class_vocab_framed_pct']}%) | NEEDS_CONTEXT {ov['NEEDS_CONTEXT_pct']}%")

    by = Counter((f["severity"], f["check"]) for f in flags)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    print("\nFlags:" + ("  none" if not by else ""))
    for (sev, chk), cnt in sorted(by.items(), key=lambda kv: (order[kv[0][0]], -kv[1])):
        ex = [f for f in flags if f["severity"] == sev and f["check"] == chk][:3]
        rows = ",".join((f"{f.get('phase')}:{f['row_id']}" if phase.startswith("ALL") else str(f["row_id"]))
                        for f in ex if f["row_id"] != "")
        print(f"  [{sev:5}] {chk:34} x{cnt:<5} rows: {rows or '-':14} {ex[0]['detail'][:70] if ex else ''}")

    if not show_dist or not ov["coded"]:
        return
    rates = [x for x in long if x["type"] == "binary"]
    print("\nBinary code prevalence (% of coded rows" + (", weighted" if ov.get("weighted") else "") + "):")
    line = [f"{x['variable']}={x['n']}({x['pct_of_coded']}%)" for x in rates]
    for i in range(0, len(line), 4):
        print("  " + "  ".join(line[i:i + 4]))
    print("\nCategorical and multi-label distributions (n, % of coded rows):")
    for var in list(SCALARS) + list(LISTS):
        cats = [x for x in long if x["variable"] == var]
        if cats:
            body = ", ".join(f"{x['category']} {x['n']} ({x['pct_of_coded']}%)" for x in cats)
            extra = f"  [labels/row {ov[f'{var}_labels_per_row']}]" if f"{var}_labels_per_row" in ov else ""
            print(f"  {var}: {body}{extra}")


def print_cross_phase(overviews: list[dict]) -> None:
    cols = (["phase", "rows", "coded", "coverage_pct", "rows_flagged_pct", "words_mean"]
            + [f"{c}_pct" for c in KEY_RATES]
            + ["STANCE_CRITICAL_pct", "STANCE_DEFENDING_pct", "STANCE_NEUTRAL_pct",
               "NEEDS_CONTEXT_pct", "CLASS_FRAME_pct", "class_vocab_framed_pct",
               "class_vocab_uncoded_pct", "uncoded_pct_words_gt80"])
    table = pd.DataFrame(overviews)
    table = table[[c for c in cols if c in table.columns]]
    table.columns = [c.replace("_pct", "%").replace("STANCE_", "ST_") for c in table.columns]
    print(f"\n{'=' * 78}\nACROSS PHASES (code rates are % of coded rows)\n{'=' * 78}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table.to_string(index=False))


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="explicit results CSVs or directories (overrides --phases)")
    ap.add_argument("--crisis", default="ateneo-drowning", help="crisis slug under the output root")
    ap.add_argument("--phases", default="all", help="'all' or comma-separated phases, e.g. 01-onset,04-reckoning")
    ap.add_argument("--pass-dir", default="full-codebook-coding", help="pass directory under the crisis")
    ap.add_argument("--output-root", type=Path, default=HERE.parent.parent / "output", help="run-class-emotions.py output root")
    ap.add_argument("--no-distributions", action="store_true", help="skip per-phase distribution printouts")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as ERROR")
    args = ap.parse_args()

    targets = discover(args)
    if not targets:
        print("No full_codebook_results.csv files found.")
        return 2

    overviews: list[dict] = []
    long_all: list[dict] = []
    flags_all: list[dict] = []
    frames: list[pd.DataFrame] = []
    for phase, path in targets:
        df, flags = validate(path)
        for f in flags:
            f["phase"] = phase
        out = path.with_name(path.stem + "_validation_flags.csv")
        pd.DataFrame(flags, columns=FLAG_COLS).to_csv(out, index=False)
        ov, long = describe(phase, df, flags, path)
        print_phase(phase, path, ov, long, flags, not args.no_distributions)
        print(f"\nFlags written -> {out}")
        overviews.append(ov)
        long_all.extend(long)
        flags_all.extend(flags)
        frames.append(df.assign(phase=phase))

    # Combined statistics across the selected phases.
    if len(targets) > 1:
        combined = pd.concat(frames, ignore_index=True)
        ov_all, long_comb = describe("ALL", combined, flags_all, None)
        for key in ("hard_failures", "pending"):
            ov_all[key] = sum(o.get(key) or 0 for o in overviews)
        reasons_all: Counter = Counter()
        for o in overviews:
            for part in (o.get("failure_reasons") or "").split(";"):
                if ":" in part:
                    k, v = part.rsplit(":", 1)
                    reasons_all[k] += int(v)
        ov_all["failure_reasons"] = ";".join(f"{k}:{v}" for k, v in reasons_all.most_common())
        versions = {v.split(":")[0] for o in overviews for v in (o.get("prompt_versions") or "").split(";") if v}
        models = {v.split(":")[0] for o in overviews for v in (o.get("models") or "").split(";") if v}
        ov_all["prompt_versions"] = ";".join(sorted(versions))
        ov_all["models"] = ";".join(sorted(models))
        precs = {v.split(":")[0] for o in overviews for v in (o.get("precisions") or "").split(";") if v}
        ov_all["precisions"] = ";".join(sorted(precs))
        print_phase("ALL SELECTED PHASES", Path(", ".join(p for p, _ in targets)), ov_all, long_comb,
                    flags_all, not args.no_distributions)
        overviews.append(ov_all)
        long_all.extend(long_comb)
        print_cross_phase(overviews)
    else:
        versions = {v.split(":")[0] for v in (overviews[0].get("prompt_versions") or "").split(";") if v}
        models = {v.split(":")[0] for v in (overviews[0].get("models") or "").split(";") if v}
        precs = {v.split(":")[0] for v in (overviews[0].get("precisions") or "").split(";") if v}

    # Results from different prompt versions or models must not be pooled.
    if len(versions) > 1:
        print(f"\nWARNING: results mix prompt versions {sorted(versions)}; do not pool them. "
              "Recode older phases with the current prompt.")
        flags_all.append({"phase": "ALL", "row_id": "", "comment_id": "", "severity": "WARN",
                          "check": "mixed_prompt_versions", "detail": str(sorted(versions))})
    precs.discard("unrecorded")
    if len(precs) > 1:
        print(f"\nWARNING: results mix precisions {sorted(precs)} (e.g. bf16 vs fp8 vs nf4); "
              "do not pool them without disclosure.")
        flags_all.append({"phase": "ALL", "row_id": "", "comment_id": "", "severity": "WARN",
                          "check": "mixed_precision", "detail": str(sorted(precs))})
    if len(models) > 1:
        print(f"\nWARNING: results mix models {sorted(models)}; do not pool them.")
        flags_all.append({"phase": "ALL", "row_id": "", "comment_id": "", "severity": "WARN",
                          "check": "mixed_models", "detail": str(sorted(models))})

    if not args.paths:
        # One folder per selection, so a single-phase check never overwrites the
        # all-phases summary.
        scope = "all" if args.phases.strip().lower() == "all" else "+".join(p for p, _ in targets)
        summary_dir = args.output_root / args.crisis / args.pass_dir / "_validation" / scope
        summary_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flags_all, columns=FLAG_COLS).to_csv(summary_dir / "flags_all.csv", index=False)
        pd.DataFrame(overviews).to_csv(summary_dir / "phase_overview.csv", index=False)
        pd.DataFrame(long_all).to_csv(summary_dir / "stats_long.csv", index=False)
        print(f"\nSummary tables written -> {summary_dir}/ (flags_all, phase_overview, stats_long)")

    errs = sum(f["severity"] == "ERROR" for f in flags_all)
    warns = sum(f["severity"] == "WARN" for f in flags_all)
    print(f"\nOverall: {errs} ERROR flags, {warns} WARN flags across {len(targets)} phase(s)")
    return 1 if errs or (args.strict and warns) else 0


if __name__ == "__main__":
    sys.exit(main())
