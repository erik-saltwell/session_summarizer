# update-session-settings

Use this skill when adding a new setting to `SessionSettings` or updating an existing one.

## Files to update

There are four places that must all be kept in sync:

1. `src/session_summarizer/protocols/session_settings.py` — the Pydantic model
2. Validation rules — inside the same file, in `_validate_audio_file` (model validator) or a new `@field_validator`
3. `src/session_summarizer/console/main.py` — the `_SAMPLE_SETTINGS` string
4. `data/settings.yaml` — the live default settings file

---

## Step 1 — Add the field to `SessionSettings`

In `session_settings.py`, add the field to the `SessionSettings` class body.

Rules:
- All fields are **required** (no `default=` or `Optional`). Use `Annotated[<type>, Field(...)]`.
- Place the field near logically related fields.
- The `Field(description=...)` must be a clear, one-sentence explanation of what the setting controls.

Example pattern:
```python
my_new_setting: Annotated[
    float,
    Field(description="One-sentence description of what this controls"),
]
```

For nested config (sub-model), define a new frozen `BaseModel` above `SessionSettings` and use it as the field type, following the pattern of `VadSettings`.

---

## Step 2 — Add validation

Add validation in `session_settings.py`. Choose the right validator type:

- **`@field_validator("field_name")`** — for single-field checks (range, allowed values, non-empty strings, etc.)
- **`@model_validator(mode="after")`** — for cross-field checks (e.g. min < max relationships)

Existing validators to extend if applicable:
- `_validate_audio_file` (model validator) already checks min/max pairs — add similar checks there for new paired min/max fields.

Raise `ValueError` with a descriptive message that names the field and explains the constraint.

---

## Step 3 — Update `_SAMPLE_SETTINGS` in `main.py`

In `src/session_summarizer/console/main.py`, add a new block to the `_SAMPLE_SETTINGS` string.

Follow the established style exactly:

```yaml
# ---------------------------------------------------------------------------
# <field_name>  (REQUIRED)
# ---------------------------------------------------------------------------
# What this setting does — one or two sentences explaining its purpose and
# effect on processing.
#
# Allowed values:
#   <value1>  — explanation
#   <value2>  — explanation
#
# Default: <default_value>
# Reasonable range: <min>–<max>  (if numeric)
#
# Example:
#   <field_name>: <example_value>
<field_name>: <default_value>
```

Place the block near logically related settings (e.g. a new segment length goes near the other `min/max_segment_length_*` blocks).

---

## Step 4 — Update `data/settings.yaml`

Open `data/settings.yaml` and add the same block you wrote for `_SAMPLE_SETTINGS`. The two files must stay identical in content — `data/settings.yaml` is the live copy of the sample.

Place the new block in the same relative position as in `_SAMPLE_SETTINGS`.

---

## Checklist

Before finishing, verify:

- [ ] Field added to `SessionSettings` with `Annotated[<type>, Field(description=...)]`
- [ ] No `default=` or `Optional` — the field is required
- [ ] Validation added (field validator or model validator)
- [ ] `_SAMPLE_SETTINGS` block added with full documentation header, allowed values, default, and example
- [ ] `data/settings.yaml` updated to match `_SAMPLE_SETTINGS` exactly
- [ ] If it is a `Path` field that stores a file path, add the key to the `_resolve_paths` method so relative paths are resolved correctly
