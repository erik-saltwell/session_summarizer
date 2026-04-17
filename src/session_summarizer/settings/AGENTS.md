
Use these steps when adding, removing or updating session settings. Settings are stored in nested, hierarchical `settings.yaml` files and loaded via `SessionSettings.load_cascading(session_id)`, which merges `data/settings.yaml` with an optional per-session override from `<session_dir>/settings.yaml`.

## Files to update

There are four places that must be kept in sync:

1. **The pydantic model.** `SessionSettings` lives in `src/session_summarizer/settings/session_settings.py` (declared `class SessionSettings(BaseModel, frozen=True)`). Nested sub-models also live in `session_settings.py` (e.g. `PipelinePaths`, `SegmentationSettings`, `SpeakerClipSettings`, `SpeakerIdentificationSettings`, `SessionInfo`, `CampaignInfo`, `GlossaryEntry`) — or in sibling modules for larger groupings (`vad_settings.py`, `diarization_stitching_settings.py`), in which case they are re-exported from `settings/__init__.py`.
2. **Validation rules** on the pydantic models — field validators for single-field checks and model validators for cross-field checks.
3. **`_SAMPLE_SETTINGS`** in `src/session_summarizer/console/main.py` — the template consumed by the `generate-sample-settings` CLI command, which writes to `data/settings.yaml`.
4. **`data/settings.yaml`** — the committed base-settings file. It must mirror `_SAMPLE_SETTINGS` exactly (the `generate-sample-settings` command overwrites it from `_SAMPLE_SETTINGS`).

---

## Step 1 — Add the field to `SessionSettings`

In `session_settings.py`, add the field to the `SessionSettings` class body or one of the nested classes.

Rules:
- New fields are **required** (no `default=`, no `Optional`) — users must supply a value in `settings.yaml`. A few legacy fields such as `SpeakerClipSettings.silence_gap_seconds` carry a default for backward compatibility, but do not add new defaulted fields without a strong reason.
- Use `Annotated[<type>, Field(description=...)]` — this is the dominant pattern (`SessionSettings`, `PipelinePaths`, `SegmentationSettings`, `SpeakerClipSettings`, etc.). The older sub-models `VadSettings` and `DiarizationStitchingSettings` use bare `Field(...)` on typed attributes; don't copy that style for new code.
- Place the field near logically related fields, and group related settings into a nested sub-model rather than flattening them onto `SessionSettings` directly.
- `Field(description=...)` must be a clear, one-sentence explanation of how the setting is used in code. Reference the consuming command/helper by name where it helps (e.g. `"Used by: merge_speaker_clips command."`) — existing descriptions in `SpeakerClipSettings` do this.

Example pattern:
```python
my_new_setting: Annotated[
    float,
    Field(description="One-sentence description of what this controls. Used by: <command>."),
]
```

### Nested sub-models

Define nested sub-models as frozen `BaseModel` subclasses:

```python
class MyFeatureSettings(BaseModel, frozen=True):
    """One-line purpose."""

    threshold: Annotated[float, Field(description="...")]
    ...
```

- Small, tightly-scoped sub-models (fewer than ~10 fields, used only by `SessionSettings`) go **above** `SessionSettings` in `session_settings.py`.
- Larger or independently-reused sub-models go in their own file under `settings/` (see `vad_settings.py`, `diarization_stitching_settings.py`) and must be re-exported from `settings/__init__.py`.
- Expose the sub-model on `SessionSettings` with `Annotated[MyFeatureSettings, Field(description=...)]`.

### Output paths go in `PipelinePaths`

Path fields for pipeline artifacts belong on the `PipelinePaths` sub-model (accessed as `settings.paths.<name>`), not on `SessionSettings` directly. Declare them as `Path` type with an `Annotated[..., Field(description=...)]` wrapper, matching the existing entries (`cleaned_audio`, `transcript`, `punctuated_text`, etc.). **You do not need to touch `_resolve_paths`** — it iterates every key under `paths` dynamically and converts relative YAML strings into absolute `Path` objects against the session directory.

---

## Step 2 — Add validation

Add validation on the class that owns the field (either `SessionSettings` or the relevant nested sub-model). Choose the right validator type:

- **`@field_validator("field_name")`** — for single-field checks (range, allowed values, non-empty strings, etc.). Use the multi-target form `@field_validator("field_a", "field_b")` with `ValidationInfo` when several fields share the same constraint — `SpeakerClipSettings._lead_must_be_non_negative` and `_duration_must_be_non_negative` are the canonical examples.
- **`@model_validator(mode="after")`** — for cross-field checks (e.g. min < max relationships). `SegmentationSettings._validate_bounds` is the canonical example.

Existing validators to extend when applicable:
- `PipelinePaths._validate_audio_suffix` (field validator on `source_audio`) — extend if you add another path that must enforce a file-extension constraint.
- `SegmentationSettings._validate_bounds` — extend for any new min/max pair of segment-length fields.
- The range validators in `SpeakerClipSettings` (`_similarity_residual_must_be_in_range`, `_min_centroid_similarity_must_be_in_range`) — add the new field to an existing validator's decorator list if it shares the same constraint, rather than writing a new one.

Raise `ValueError` with a descriptive message that names the field and explains the constraint. Use the `info.field_name` from `ValidationInfo` when a validator covers multiple fields so the error identifies which one failed.

---

## Step 3 — Update `_SAMPLE_SETTINGS` in `main.py`

In `src/session_summarizer/console/main.py`, add a new block to the `_SAMPLE_SETTINGS` string. This constant is written to `data/settings.yaml` by the `generate-sample-settings` CLI command, so treat it as the authoritative documentation surface for the setting.

### Top-level scalar or a fresh nested section

Use the banner-comment style for top-level fields and the headers of nested sections:

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
# Reasonable range: <min>–<max>  (if numeric)
#
# Example:
#   <field_name>: <example_value>
<field_name>: <example_value>
```

### Field inside an already-declared nested section

Inside a nested section like `speaker_clips:` or `segmentation:`, use the shorter inline-comment style that the existing fields in those sections use — a short purpose sentence, a `Used by:` pointer, and `Allowed values:` / reasonable default on one line:

```yaml
speaker_clips:

  # Seconds of audio padding before each speaker clip.
  # Used by: create_speaker_clips command.
  # Allowed values: >= 0.0 (seconds). Reasonable default: 0.25
  lead_in_seconds: 0.25
```

Place the block near logically related settings (e.g. a new segment length goes near the other `min/max_segment_length_*` blocks).

After editing `_SAMPLE_SETTINGS`, regenerate `data/settings.yaml` so the two stay identical. The simplest way is `uv run session-summarizer generate-sample-settings`; otherwise copy the string verbatim.

---

## Checklist

Before finishing, verify:

- [ ] Field added to `SessionSettings` (or the appropriate nested sub-model) with `Annotated[<type>, Field(description=...)]`
- [ ] No `default=`, no `Optional` — the field is required (new defaulted fields need a strong justification)
- [ ] Nested sub-model is `class X(BaseModel, frozen=True)`; if it lives in its own file, it's re-exported from `settings/__init__.py`
- [ ] Validation added where applicable — extended an existing validator when the constraint matches (range, non-negative, min/max bound); otherwise added a new `@field_validator` or `@model_validator(mode="after")`
- [ ] `_SAMPLE_SETTINGS` block added using the banner style for top-level/new-section fields or the inline-comment style for fields inside an existing section — purpose, `Used by:`, allowed values / reasonable default, and an `Example:` where useful
- [ ] `data/settings.yaml` regenerated to match `_SAMPLE_SETTINGS` (run `generate-sample-settings` or copy verbatim)
- [ ] For new `Path` fields: put them on `PipelinePaths` so `_resolve_paths` picks them up automatically — no code change to `_resolve_paths` needed. If a path field must live outside `paths:`, resolve it explicitly.
- [ ] `uv run pyright src tests` clean after adjustments
