# Using your own data

You can run the pipeline on any crisis you choose, as long as the data follows the format below.
Each crisis gets its own folder, `00-data/{crisis-slug}/`, and every stage selects it with the
`CRISIS` environment variable. The committed `ateneo-drowning/` folder is a complete working
example to copy from.

**In short:**

1. Pick a slug and put your data in `00-data/{slug}/` in the format below.
2. Run `python3 00-data/check_dataset.py {slug}` and fix every `ERROR`.
3. Make the small per-crisis edits in [What to change for a new crisis](#what-to-change-for-a-new-crisis).
4. Run the stages with `CRISIS={slug}`.

## 1. Choose a slug and assign phases

- **Slug:** lowercase kebab-case, describing the event, e.g. `flood-response` or
  `ateneo-drowning`. It becomes the folder name everywhere: `00-data/`, every stage's `output/`,
  and `phase-context.json`.
- **Phases:** the pipeline always uses the same five phase folders. You decide which date range
  and which videos go into each one, using the trigger rules in
  [`CRISIS-PHASES-TEMPLATE.md`](CRISIS-PHASES-TEMPLATE.md):

  | Folder | Phase | Ends when |
  |---|---|---|
  | `01-onset` | Onset | exactly 24 h after the incident |
  | `02-confirmation` | Confirmation | the first formal investigation opens, or 72 h, whichever comes first |
  | `03-spread` | Spread | the responsible party's first formal accountability act |
  | `04-reckoning` | Reckoning | that party's last accountability act |
  | `05-lingering` | Lingering | a cutoff date you fix before collecting |

  Keep these exact folder names. A phase you have no data for can stay empty, and the pipeline
  skips it. Renaming phases or adding a sixth means editing the `PHASES` lists in every stage.
- **Assign by content, not by upload date.** A video goes in the phase whose events it covers.
  Write down the anchor event for each phase; you'll need it for `phase-context.json` and the
  methods section.

## 2. Folder layout

```
00-data/{slug}/
├── youtube-comments/                 REQUIRED
│   ├── 01-onset/
│   │   ├── {video_id}_comments.csv   one CSV per video
│   │   └── ...
│   ├── 02-confirmation/
│   ├── 03-spread/
│   ├── 04-reckoning/
│   └── 05-lingering/
├── reddit-data/                      optional
│   ├── 00-unrelated/                 optional control set: threads from the same subreddits, not about the crisis
│   ├── 01-onset/
│   │   └── {subreddit}/
│   │       └── {any-name}.csv        one CSV per thread
│   └── ...
└── video-transcripts/                optional
    └── transcripts-{name}/
        └── timestamped.vtt           one folder per video
```

Only YouTube comments are required. Stages 03–04 (LLM coding, findings) use only the
YouTube comments. Reddit and transcripts go through stages 01 and 02 only.

## 3. File formats

All files must be **UTF-8**. Timestamps must be **ISO 8601**, preferably with a timezone, e.g.
`2026-06-08T15:18:02+00:00`. Extra columns are fine and are ignored.

### YouTube comments: `youtube-comments/{phase}/{video_id}_comments.csv`

One row per comment, with replies included as their own rows.

| Column | Required | Type / format | Notes |
|---|---|---|---|
| `video_id` | yes | string | YouTube video ID. Also used to find spam (the same text posted on 3+ videos) |
| `comment_id` | yes | string, unique across the whole crisis | |
| `parent_comment_id` | yes | string | `root` for a top-level comment, otherwise the parent's `comment_id` |
| `text` | yes | string | Original comment text, uncleaned. Preprocessing cleans it |
| `like_count` | yes | integer | |
| `create_time` | yes | ISO 8601 | |
| `user_handle` | yes | string, e.g. `@name` | Pseudonymized in preprocessing and never written to the outputs |
| `video_title`, `channel_id`, `user_display_name`, `reply_count`, `is_reply` | no | | Kept in the raw data for reference only. `is_reply` is ignored; replies are found through `parent_comment_id` |

Minimal example:

```csv
video_id,comment_id,parent_comment_id,text,like_count,create_time,user_handle
dQw4w9WgXcQ,Ugx1,root,Condolences to the families 🙏,12,2026-06-08T16:02:00+00:00,@viewer1
dQw4w9WgXcQ,Ugx2,Ugx1,Same. Sana may managot,3,2026-06-08T16:10:00+00:00,@viewer2
```

How you collect the comments is up to you (for example yt-dlp or the YouTube Data API). Only
the exported CSV has to match this format.

### Reddit threads: `reddit-data/{phase}/{subreddit}/{thread}.csv`

One file per thread: one `post` row followed by its `comment` rows.

| Column | Required | Notes |
|---|---|---|
| `type` | yes | `post` or `comment` |
| `id` | yes | Reddit fullname, e.g. `t3_1u0awsp` for the post |
| `text` | yes | For the post, the first line is used as the thread title |
| `author`, `author_fullname`, `replying_to_user`, `metadata_json` | yes, may be empty | Read for anonymization, then dropped |
| `score` | yes | integer |
| `created_time` | yes | ISO 8601 |
| `subreddit` | yes | |

Optional: `00-data/{slug}/reddit-data/phase-mapping.csv` (`post,date,subreddit,n_comments,phase,title`),
a record of which thread went into which phase. The pipeline doesn't read it, but keep it for
the methods section.

### Video transcripts: `video-transcripts/transcripts-{name}/timestamped.vtt`

A standard WebVTT file (YouTube's auto-captions export works) with `HH:MM:SS.mmm --> HH:MM:SS.mmm`
cues. `{name}` is a short ID such as `ateneo_presscon`; you'll add it to three run scripts (see
below).

## 4. Check the format

```bash
python3 00-data/check_dataset.py {slug}
```

The checker uses only the standard library, so it runs before any setup. It verifies:
- the folder names
- the required columns
- UTF-8 encoding
- ISO timestamps
- unique comment IDs
- that replies point to comments in the data
- that each Reddit file has one post
- the VTT cues
- the stage 03 phase context
- the transcript lists in the run scripts

`ERROR` means preprocessing will fail or give wrong results, so fix those first. `WARN` is worth
reading but won't stop the run. For example, `ateneo-drowning` has 119 replies whose parent
comment wasn't collected.

## What to change for a new crisis

Everything else reads the slug from `CRISIS`.

| Stage | File | Change | Needed? |
|---|---|---|---|
| 01 | `01-preprocessing/notebooks/youtube-comments-preprocessing.ipynb` | `_INSTITUTIONAL`: public accounts (the organization, news channels) to keep un-pseudonymized | Recommended |
| 01, 02 | `01-preprocessing/run_transcripts.sh`, `02-analysis/transcripts/run_sentiment.sh`, `02-analysis/transcripts/run_topics.sh` | Put your transcript `{name}`s in the `TRANSCRIPTS=( … )` list | Only with transcripts |
| 02 | `02-analysis/youtube/youtube-comments-sentiment-change-across-phases.ipynb` | Phase event labels (date and key events per phase) and the `"Date (June 2026)"` axis label | Recommended (figure labels only) |
| 03 | `03-llm-passes/prompts/phase-context.json` | Add a `"{slug}"` entry with one neutral CASE CONTEXT sentence or two per phase: when the videos were posted and what was publicly known. No judgments about fault. Copy the `ateneo-drowning` entry's style | Required for stage 03 |
| 03 | `03-llm-passes/prompts/*/…-prompt-v*.txt` | The few-shot examples come from the Ateneo case. They still work for other Filipino crises; for a very different case, write a new prompt version (see the versioning rules in `03-llm-passes/README.md`) | Optional |
| 04 | `04-statistical-analysis/notebooks/*.ipynb` | `CRISIS = "ateneo-drowning"` is set at the top of each notebook; change it to your slug. In `01-merge-and-prep`, point `ARCHIVE` at your stage 03 output (`03-llm-passes/output`) | Required for stage 04 |

Then run the pipeline:

```bash
export CRISIS={slug}                      # or put CRISIS={slug} before each command
bash setup.sh                             # once per machine
python3 main.py --stage preprocess        # 01 → 01-preprocessing/output/{slug}/
python3 main.py --stage analysis          # 02 → 02-analysis/output/{slug}/ (needs GEMINI_API_KEY)
# 03: on a GPU machine, see 03-llm-passes/README.md (use --crisis {slug})
# 04: open the notebooks (see the main README)
```

The run scripts and `main.py` read `CRISIS` from the shell, not from `.env`.

## Language and privacy notes

- **Language.** The models are multilingual (XLM-RoBERTa sentiment, multilingual MiniLM
  embeddings). The Tagalog spaCy model, the codebook and the Filipino-emotions pass assume
  Filipino/Taglish comments. Data in other languages runs through stages 01–02, but stage 03's
  codes were designed and validated for Filipino discourse.
- **Privacy.** Preprocessing pseudonymizes handles (SHA-256) and redacts @mentions, so the outputs
  contain no usernames. The **raw** files in `00-data/` still contain them. This repo commits its
  raw data. If you don't want yours public, add `00-data/{slug}/` to `.gitignore`.
