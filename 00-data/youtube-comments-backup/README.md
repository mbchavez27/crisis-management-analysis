# YouTube Comments Dataset

Comments from major YouTube videos covering the Ateneo drowning crisis and its
institutional aftermath (June 10–23, 2026).

## Sampling Frame

- **Source:** Major Philippine news networks (ANC/ABS-CBN, GMA, Rappler),
  expert commentary shows (On Point, The Scorecard), and independent creators
- **Inclusion:** Each video assigned to the chronological crisis phase its
  content addresses
- **Scale:** 58 videos across 9 phases, 10,836 raw comments
- **No minimum engagement threshold** applied; comment counts range from 1 to 356

## Directory Structure

```
├── 01-the_incident/            (7 videos)
├── 02-autopsy_request/         (8 videos)
├── 03-public_outcry/           (6 videos)
├── 03.1-baldwin_apology/       (5 videos)
├── 04-scholarships_withdrawal/ (4 videos)
├── 05-resignation_presscon/    (7 videos)
├── 06-townhall_podcast/        (5 videos)
├── 07-uaap_ban/                (4 videos)
└── 08-legal_probe/             (12 videos)
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
