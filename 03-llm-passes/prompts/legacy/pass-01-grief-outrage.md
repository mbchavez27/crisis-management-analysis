# Pass 1: Grief & Outrage + Context

## Purpose

Detect grief/outrage in YouTube comments and extract the invoked value/relationship plus whose family is referenced.

## Input columns

CSV from `df.to_csv(index=False)` with columns:
`video_id`, `text`, `like_count`, `create_time`, `is_shouting`, `sentences`

## System Prompt

```plaintext
System Instruction:
I will provide a dataset of comments in CSV format containing `video_id`, `text`, `like_count`, `create_time`, `is_shouting`, and `sentences` columns. You must return the analysis strictly as a JSON array of objects. Do not include markdown formatting (other than the JSON block) or explanations outside of the JSON.

For each comment expressing grief or outrage in the text column, extract:
(a) the emotion (specifically whether it is Grief, Outrage, or both).
(b) the value or relationship invoked as the grounds for the emotion.
(c) whose family is referenced (e.g., the victims' family, the commenter's own family, or any parent).

If a comment does not express grief or outrage, return empty values for these fields.

Use this exact JSON structure for each object:
- `video_id`: The video_id from the CSV.
- `text`: The text of the comment from the CSV.
- `grief_or_outrage`: An array of strings (e.g., ["Grief", "Outrage"]). If neither, output an empty array.
- `value_or_relationship_invoked`: A string explaining the grounds. If none, state "None".
- `family_referenced`: An array of strings specifying the family. If none, return an empty array.

Data:
[Insert your df.to_csv(index=False) output here]
```

## Output schema

| Field | Type | Notes |
|---|---|---|
| `video_id` | string | Passthrough from CSV |
| `text` | string | Passthrough from CSV |
| `grief_or_outrage` | string[] |Subset of `["Grief", "Outrage"]`; `[]` if neither |
| `value_or_relationship_invoked` | string | Grounds for emotion; `"None"` if none |
| `family_referenced` | string[] | e.g. victims' family, commenter's own family, any parent; `[]` if none |

## Notebook usage

```python
from pathlib import Path

def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("```plaintext")[1].split("```")[0].strip()

prompt_template = load_prompt(Path("../prompts/pass-01-grief-outrage.md"))
full_input = prompt_template.replace(
    "[Insert your df.to_csv(index=False) output here]",
    df.to_csv(index=False),
)
```
