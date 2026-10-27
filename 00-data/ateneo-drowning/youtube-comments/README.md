# YouTube Comments Dataset

Comments from major YouTube videos covering the Ateneo drowning crisis and its
institutional aftermath (events from June 8 to September 2026).

## Phase Definitions

5 phases, labelled Phase 1–5 in the findings charts and the paper (Table 1).
Periods are event dates from news reports, not upload dates, so Phases 3 and 4
overlap.

| # | Folder | Period (2026) | Defining Events |
|---|--------|---------------|-----------------|
| 1 | `01-onset` | June 8–9 | Two student-athletes drown during a team-building activity; police initially call it an accident |
| 2 | `02-confirmation` | June 9–11 | Families seek answers, and initial police findings and university statements are released |
| 3 | `03-spread` | June 11–15 | Public mourning, support for the victims' families, and calls for justice grow |
| 4 | `04-reckoning` | June 12–25 | The head coach apologizes and resigns, the university apologizes, and the CIDG and NBI investigate |
| 5 | `05-lingering` | June 26–September 20 | Anti-hazing charges are recommended, the UAAP permanently bans the former head coach, and the victims are honored |

## Sampling Frame

- **Source:** Major Philippine news networks (ANC/ABS-CBN, GMA, Rappler),
  expert commentary shows (On Point, The Scorecard), and independent creators
- **Inclusion:** Each video assigned to the crisis phase its content addresses,
  not the phase in which it was uploaded
- **Scale:** 64 videos across 5 phases, 11,902 raw comments (3,573 replies)
- **No minimum engagement threshold** applied; comment counts range from 1 to 4,255

## Directory Structure

```
├── 01-onset/          (7 videos, 907 comments)
├── 02-confirmation/   (14 videos, 2,698 comments)
├── 03-spread/         (8 videos, 5,505 comments)
├── 04-reckoning/      (28 videos, 1,439 comments)
└── 05-lingering/      (7 videos, 1,353 comments)
```

Preprocessing removes spam, duplicates, and comments with no meaningful text,
leaving 11,561 comments.

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
