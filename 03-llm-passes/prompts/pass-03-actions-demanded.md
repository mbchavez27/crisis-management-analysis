# Pass 3: Actions Demanded

## Purpose

Identify the action called for or demanded in each comment.

## Input columns

CSV from `df.to_csv(index=False)` with columns:
`video_id`, `text`, `like_count`, `create_time`, `is_shouting`, `sentences`

## System Prompt

```plaintext
System Instruction:
I will provide a dataset of comments in CSV format containing `video_id`, `text`, `like_count`, `create_time`, `is_shouting`, and `sentences` columns. You must return the analysis strictly as a JSON array of objects. Do not include markdown formatting (other than the JSON block) or explanations outside of the JSON.

For each comment's text, identify the action that is being called or demanded.

Use this exact JSON structure for each object:
- `video_id`: The video_id from the CSV.
- `text`: The text of the comment from the CSV.
- `action_demanded`: A string detailing the action called for. If no explicit action is demanded, state "None" or describe the implicit demand.

Data:
[Insert your df.to_csv(index=False) output here]
```

## Output schema

| Field | Type | Notes |
|---|---|---|
| `video_id` | string | Passthrough from CSV |
| `text` | string | Passthrough from CSV |
| `action_demanded` | string | Explicit demand; `"None"` or implicit demand if none explicit |

## Notebook usage

```python
from pathlib import Path

def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("```plaintext")[1].split("```")[0].strip()

prompt_template = load_prompt(Path("../prompts/pass-03-actions-demanded.md"))
full_input = prompt_template.replace(
    "[Insert your df.to_csv(index=False) output here]",
    df.to_csv(index=False),
)
```
