# Crisis Phases Template

Canonical reference for organizing data into crisis phases. Every new crisis
added to this dataset must follow this structure.

## Theoretical Framework

Phases are grounded in **Fink's (1986) four-stage crisis management model** with
the chronic stage split into onset/response sub-phases, and **Coombs & Tachkova's
(2024) sticky crisis** concept for the long-tail phase.

| # | Phase Name | Folder Slug | Theoretical Basis | Start Trigger | End Trigger |
|---|-----------|-------------|-------------------|---------------|-------------|
| 1 | Onset | `01-onset` | Fink (1986), Acute stage (sub-phase 1) | Incident occurs | Exactly +24h from incident time (fixed clock) |
| 2 | Confirmation | `02-confirmation` | Fink (1986), Acute stage (sub-phase 2) | +24h mark | First formal investigative/regulatory body opens, OR 72h — whichever comes first |
| 3 | Spread | `03-spread` | Fink (1986), Chronic stage (onset) | Phase 2 end trigger fires | First formal accountability act by responsible party |
| 4 | Reckoning | `04-reckoning` | Fink (1986), Chronic stage (response) | First accountability act | Last accountability act attributable to same responsible party |
| 5 | Lingering | `05-lingering` | Coombs & Tachkova (2024), sticky crisis | End of Phase 4 | Pre-registered cutoff (fixed calendar window) |

### Phase Descriptions

**Phase 1 — Onset**  
First-wave shock and grief before any official account exists. The incident
has just occurred; public reaction is raw, emotional, and information-sparse.
Duration is exactly 24 hours (fixed clock, not content-dependent).

**Phase 2 — Confirmation**  
Official framing lands — statements, casualty counts, initial institutional
positions. Reaction is still purely reactive. 72h is a ceiling for slow cases,
not a floor that overrides fast ones.

**Phase 3 — Spread**  
Secondary storylines, rumor, blame-widening, media debate. The crisis has
escaped the initial frame. Multiple narratives compete. Ends when the
responsible party takes a formal accountability act.

**Phase 4 — Reckoning**  
The organization's own internal reckoning — apologies, resignations, internal
sanctions. Ends when the responsible party's last accountability act occurs,
before external/regulatory bodies take over adjudicating.

**Phase 5 — Lingering**  
External/regulatory apparatus, legal proceedings, periodic re-triggering,
long-tail attention decay. The crisis has become "sticky" — it may resurface
indefinitely but at lower intensity.

---

## JSON Schema

```json
{
  "phases": [
    {
      "id": 1,
      "name": "Onset",
      "slug": "01-onset",
      "basis": "Fink (1986), Acute stage (sub-phase 1)",
      "start_trigger": "Incident occurs",
      "end_trigger": "Exactly +24h from incident time (fixed clock, not content-dependent)",
      "duration_hours": 24
    },
    {
      "id": 2,
      "name": "Confirmation",
      "slug": "02-confirmation",
      "basis": "Fink (1986), Acute stage (sub-phase 2)",
      "start_trigger": "+24h mark",
      "end_trigger": "First formal investigative/regulatory body opens, OR 72h — whichever first",
      "duration_hours": "variable (up to 48h)"
    },
    {
      "id": 3,
      "name": "Spread",
      "slug": "03-spread",
      "basis": "Fink (1986), Chronic stage (onset)",
      "start_trigger": "Phase 2 end trigger fires",
      "end_trigger": "First formal accountability act by responsible party",
      "duration_hours": "variable"
    },
    {
      "id": 4,
      "name": "Reckoning",
      "slug": "04-reckoning",
      "basis": "Fink (1986), Chronic stage (response)",
      "start_trigger": "First accountability act",
      "end_trigger": "Last accountability act attributable to same responsible party",
      "duration_hours": "variable"
    },
    {
      "id": 5,
      "name": "Lingering",
      "slug": "05-lingering",
      "basis": "Coombs & Tachkova (2024), sticky crisis",
      "start_trigger": "End of Phase 4",
      "end_trigger": "Pre-registered cutoff (fixed calendar window)",
      "duration_hours": "variable"
    }
  ]
}
```

---

## Case: Ateneo Drowning (2026)

| # | Phase | Dates | Key Events |
|---|-------|-------|------------|
| 1 | Onset | Jun 8, 3:45pm → Jun 9, 3:45pm | Drowning of Rene Baterbonia and Divine Adili; initial shock; Ateneo confirms deaths |
| 2 | Confirmation | Jun 9, 3:45pm → Jun 11 | Family requests autopsy; PNP Aurora says no foul play; Ateneo coordinates with NBI/CIDG |
| 3 | Spread | Jun 11 → Jun 15 | Viral rumors (ankle weights, hazing); public outcry; Esguerra podcast; calls for probe |
| 4 | Reckoning | Jun 15 → Jun 19 | Baldwin resignation accepted; Fr. Yap statement; UAAP bans coaching staff |
| 5 | Lingering | Jun 19 → cutoff | CIDG/NBI hearings; hazing investigation; legal proceedings |

---

## Directory Structure Per Crisis

```
00-data/{crisis-slug}/
├── youtube-comments/
│   ├── 01-onset/
│   ├── 02-confirmation/
│   ├── 03-spread/
│   ├── 04-reckoning/
│   └── 05-lingering/
├── reddit-data/
│   ├── 00-unrelated/
│   ├── 01-onset/
│   ├── 02-confirmation/
│   ├── 03-spread/
│   ├── 04-reckoning/
│   ├── 05-lingering/
│   └── phase-mapping.csv
└── video-transcripts/
    └── transcripts-{source}/
        └── timestamped.vtt
```

---

## Adding a New Crisis

File formats, the per-crisis edits and the format checker are in
[`README.md`](README.md) (*Using your own data*). Check a new dataset with
`python3 00-data/check_dataset.py {slug}`.

---

## Phase Assignment Rules

- **Content-based:** Assign by what the video/thread discusses, not when it was posted
- **Anchor event:** Identify the key event that places content in a phase
- **Boundary cases:** When an event straddles two phases, assign to the phase where
  the majority of the content's discussion falls
- **Spillover is expected:** Comments will naturally span beyond phase boundaries;
  this is normal and expected
- **No minimum engagement threshold:** Include all relevant content regardless of
  comment count

Column schemas for YouTube, Reddit and transcript files: see [`README.md`](README.md#3-file-formats).
