# YouTube Comments Dataset

Comments from major YouTube videos covering the Ateneo drowning crisis and its
institutional aftermath (June 10–23, 2026).

## Phase Definitions

5 chronological phases based on trigger-based boundaries:

| # | Phase | Trigger Boundary |
|---|-------|-----------------|
| 1 | Immediate Reaction (24h) | Incident → +24 hours exactly |
| 2 | Immediate Reaction (Extended) | +24h → first official casualty/status confirmation, or 72h, whichever is later |
| 3 | Escalation | Phase 2 end → first formal investigative/regulatory body opens |
| 4 | Institutional Response | Formal investigation opens → first formal accountability outcome |
| 5 | Aftermath / Long Tail | Formal outcome → pre-registered cutoff |

## Sampling Frame

- **Source:** Major Philippine news networks (ANC/ABS-CBN, GMA, Rappler),
  expert commentary shows (On Point, The Scorecard), and independent creators
- **Inclusion:** Each video assigned to the chronological crisis phase its
  content addresses
- **Scale:** 58 videos across 5 phases, 10,836 raw comments
- **No minimum engagement threshold** applied; comment counts range from 1 to 356

## Directory Structure

```
├── 01-immediate_reaction_24h/     (7 videos, 1,034 comments)
├── 02-immediate_reaction_extended/ (14 videos, 3,197 comments)
├── 03-escalation/                  (9 videos, 8,071 comments)
├── 04-institutional_response/      (28 videos, 1,727 comments)
└── 05-aftermath/                   (0 videos, 0 comments)
```

## Schema

Each CSV has 12 columns:

| Column | Description |
|--------|-------------|
| `video_id` | YouTube video identifier |
| `video_title` | Title of the video |
| `channel_id` | Channel ID (empty in all files) |
| `comment_id` | Unique comment identifier |
| `text` | Comment text content |
| `user_handle` | Commenter's YouTube handle (e.g., `@username`) |
| `user_display_name` | Commenter's display name (identical to `user_handle` in all rows) |
| `like_count` | Number of likes on the comment |
| `reply_count` | Reply count (empty in all files) |
| `create_time` | Comment timestamp (ISO 8601) |
| `is_reply` | Reply flag (unreliable — all values are `"no"`) |
| `parent_comment_id` | Parent comment ID (`"root"` for top-level, comment ID for replies) |

## Notes

- `channel_id` is empty in all files (not populated by the data source)
- `reply_count` is empty in all files
- `is_reply` is unreliable (all values are `"no"`) — use
  `parent_comment_id != "root"` to identify replies
- `user_handle` and `user_display_name` are identical in all rows
- Preprocessing pseudonymizes user handles and replaces inline `@username`
  mentions in comment text before export
