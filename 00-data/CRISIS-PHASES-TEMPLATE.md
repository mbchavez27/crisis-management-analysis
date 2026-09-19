# Crisis Phases Template

Canonical reference for organizing data into crisis phases. Every new crisis
added to this dataset must follow this structure.

## Phase Definitions (5 phases, trigger-based)

| # | Phase Name | Folder Slug | Trigger Boundary |
|---|-----------|-------------|-----------------|
| 1 | Immediate Reaction (24h) | `01-immediate_reaction_24h` | Incident → +24 hours exactly |
| 2 | Immediate Reaction (Extended) | `02-immediate_reaction_extended` | +24h → first official casualty/status confirmation, or 72h, whichever is later |
| 3 | Escalation | `03-escalation` | Phase 2 end → first formal investigative/regulatory body opens |
| 4 | Institutional Response | `04-institutional_response` | Formal investigation opens → first formal accountability outcome |
| 5 | Aftermath / Long Tail | `05-aftermath` | Formal outcome → pre-registered cutoff (fixed calendar window, decided before scraping) |

## Directory Structure Per Crisis

```
00-data/{crisis-slug}/
├── youtube-comments/
│   ├── 01-immediate_reaction_24h/
│   ├── 02-immediate_reaction_extended/
│   ├── 03-escalation/
│   ├── 04-institutional_response/
│   └── 05-aftermath/
├── reddit-data/
│   ├── 00-unrelated/
│   ├── 01-immediate_reaction_24h/
│   ├── 02-immediate_reaction_extended/
│   ├── 03-escalation/
│   ├── 04-institutional_response/
│   ├── 05-aftermath/
│   └── phase-mapping.csv
└── video-transcripts/
    └── transcripts-{source}/
        └── timestamped.vtt
```

## Adding a New Crisis

1. **Choose a slug** — kebab-case, descriptive (e.g., `ateneo-drowning`, `flood-response`)

2. **Create directory structure** — follow the template above under `00-data/{slug}/`

3. **Collect YouTube data** — one `{video_id}_comments.csv` per video, placed in the
   phase folder its content addresses

4. **Collect Reddit data** — one CSV per thread, organized by subreddit subfolder
   within each phase. Update `phase-mapping.csv` with post→phase assignments.

5. **Collect transcripts** (if applicable) — VTT files under `video-transcripts/`

6. **Update pipeline config** — set `CRISIS={slug}` in all shell scripts under
   `01-preprocessing/` and `02-analysis/`

7. **Run preprocessing** — `python main.py --stage preprocess`

8. **Run analysis** — `python main.py --stage analysis`

9. **Generate insights** — `python main.py --stage insights`

## Phase Assignment Rules

- **Content-based:** Assign by what the video/thread discusses, not when it was posted
- **Anchor event:** Identify the key event that places content in a phase
- **Boundary cases:** When an event straddles two phases, assign to the phase where
  the majority of the content's discussion falls
- **Spillover is expected:** Comments will naturally span beyond phase boundaries;
  this is normal and expected
- **No minimum engagement threshold:** Include all relevant content regardless of
  comment count

## Schema Reference

### YouTube Comments CSV (12 columns)

| Column | Description |
|--------|-------------|
| `video_id` | YouTube video identifier |
| `video_title` | Title of the video |
| `channel_id` | Channel ID |
| `comment_id` | Unique comment identifier |
| `text` | Comment text content |
| `user_handle` | Commenter's YouTube handle |
| `user_display_name` | Commenter's display name |
| `like_count` | Number of likes |
| `reply_count` | Reply count |
| `create_time` | Comment timestamp (ISO 8601) |
| `is_reply` | Reply flag (unreliable — use `parent_comment_id` instead) |
| `parent_comment_id` | Parent comment ID (`"root"` for top-level) |

### Reddit Thread CSV

| Column | Description |
|--------|-------------|
| `type` | `"post"` or `"comment"` |
| `subreddit` | Subreddit name |
| `title` | Thread title (posts only) |
| `text` | Content text |
| `author` | Author username |
| `score` | Upvote score |
| `created_utc` | Timestamp (UTC) |
| `permalink` | Full Reddit URL |
