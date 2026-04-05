# create-typer-command

Use this skill when creating a new Typer CLI command in this project.

A command consists of up to four layers:
1. **Domain code** (optional) — the heavy implementation using 3rd-party packages or LLM models
2. **Helper function** — caching policy + settings extraction + calls domain code; does NOT save to disk
3. **Command class** — orchestrates prerequisites (cached) → calls own helper (uncached) → saves result
4. **CLI handler** — validates session, creates logger, instantiates command, executes it

---

## Conventions

| Thing | Convention | Example |
|---|---|---|
| CLI command name | `kebab-case` | `my-new-step` |
| Command class | `PascalCaseCommand` | `MyNewStepCommand` |
| Helper function | `snake_case` | `apply_my_new_step` |
| Helper file | `src/.../helpers/<snake_case>.py` | `my_new_step.py` |
| Command file | `src/.../commands/<snake_case>.py` | `my_new_step.py` |
| Output path setting | `<snake_case>_path` | `my_new_step_path` |
| Domain module | `src/.../transcription/`, `diarization/`, etc. | see Step 2 |

---

## Step 1 — Add settings

Use the `update-session-settings` skill to add a `Path` field for the new output file to `SessionSettings`. Also add any numeric/string tuning parameters the step needs.

```
/update-session-settings  Add my_new_step_path (Path) and my_new_step_threshold (float) ...
```

This updates `session_settings.py`, `_SAMPLE_SETTINGS`, `data/settings.yaml`, and the unit tests.

---

## Step 2 — Create domain code (only when needed)

**When to create domain code:** When the helper would contain significant implementation using 3rd-party libraries (torch, transformers, nemo, pyannote, etc.) or LLM models. Simple data transformation that doesn't pull in heavy dependencies belongs directly in the helper.

**Where it lives:** Choose an existing domain folder if it fits, or create a new one:

```
src/session_summarizer/
├── transcription/     — ASR models (Canary, Whisper, Parakeet, etc.)
├── diarization/       — speaker diarization models and clip merging logic
├── vad/               — voice activity detection models
├── turn_detection/    — turn boundary detection models
├── speaker_embeddings/— speaker embedding models
├── audio/             — audio manipulation utilities
├── analysis/          — analysis/evaluation utilities
└── my_new_domain/     — create a new folder for genuinely new domain areas
```

**Domain code rules:**
- **Never imports `SessionSettings`** — settings values are extracted by the helper and passed in as plain arguments
- **May accept `LoggingProtocol`** for progress reporting and status messages
- **May accept `GpuLogger`** if GPU memory tracking is needed
- Contains the actual model loading, inference, or heavy library calls
- Organized as classes or functions — follow the pattern of the existing domain folder you're adding to

```python
# Example: src/session_summarizer/my_new_domain/my_processor.py
from __future__ import annotations

from pathlib import Path

from ..protocols import LoggingProtocol


class MyProcessor:
    def __init__(self, device: str, threshold: float) -> None:
        # load model here — no settings reference
        self._threshold = threshold
        ...

    def process(self, audio_path: Path, logger: LoggingProtocol) -> MyResult:
        # heavy implementation using 3rd-party packages
        ...
        return result
```

Export new domain classes from the folder's `__init__.py` so the helper imports cleanly.

---

## Step 3 — Create the helper

**File:** `src/session_summarizer/helpers/<snake_case_name>.py`

The helper owns the caching policy and extracts what domain code needs from settings. **It does NOT save to disk.**

```python
from __future__ import annotations

from pathlib import Path

from ..my_new_domain import MyProcessor      # domain code import (if applicable)
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


def apply_my_new_step(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,               # direct prerequisite result
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.my_new_step_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    # Extract settings values; pass as plain arguments to domain code
    processor = MyProcessor(device=settings.device, threshold=settings.my_new_step_threshold)
    result = processor.process(session_dir / settings.cleaned_audio_file, logger)

    return result
```

**Rules:**
- Cache check always targets `session_dir / settings.<output_path_field>`.
- Cache message uses `[yellow]...[/yellow]` formatting.
- Extract what domain code needs from settings here — domain code receives plain values, never `settings`.
- Return the result; do not save it.

---

## Step 4 — Create the command class

**File:** `src/session_summarizer/commands/<snake_case_name>.py`

The command calls all prerequisites with `True`, calls its own helper with `False`, then saves.

```python
from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths

from ..helpers.add_embeddings import add_embeddings
from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_diarizer import diarize_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.first_stitcher import apply_first_stitching
from ..helpers.identity_stitch import apply_identity_stitching
from ..helpers.my_new_step import apply_my_new_step          # your helper
from ..helpers.speaker_identifier import identify_speakers
from ..helpers.transcript_aligner import align_transcript
from ..helpers.update_turn_end import update_turn_end
from ..processing_results import AlignmentResult, SpeechClipSet, TranscriptionResult
from ..settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class MyNewStepCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "My New Step"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        # --- Prerequisites: all called with use_cache_if_present=True ---
        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, True, self, self.logger
        )
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, segments, True, self, self.logger)
        alignment = score_confidence(settings, session_dir, alignment, segments, True, self, self.logger)
        diarized_clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, True, self, self.logger)
        turn_clips: SpeechClipSet = update_turn_end(settings, session_dir, diarized_clips, True, self, self.logger)
        stitched_clips: SpeechClipSet = apply_first_stitching(
            settings, session_dir, turn_clips, True, self, self.logger
        )
        embedded_clips: SpeechClipSet = add_embeddings(settings, session_dir, stitched_clips, True, self, self.logger)
        identified_clips: SpeechClipSet = identify_speakers(
            settings, session_dir, embedded_clips, True, self, self.logger
        )
        id_stitched_clips: SpeechClipSet = apply_identity_stitching(
            settings, session_dir, identified_clips, True, self, self.logger
        )

        # --- Own step: called with use_cache_if_present=False ---
        new_clips: SpeechClipSet = apply_my_new_step(
            settings, session_dir, id_stitched_clips, False, self, self.logger
        )

        # --- Save result ---
        new_clips.save_to_json(session_dir / settings.my_new_step_path)
```

### Prerequisite chain (only call what you need — stop at your direct input)

```
clean_audio
  └─ compute_vad_segments
       └─ transcribe_from_cleaned_audio
            └─ align_transcript
                 └─ score_confidence
                      └─ diarize_audio
                           └─ update_turn_end
                                └─ apply_first_stitching
                                     └─ add_embeddings
                                          └─ identify_speakers
                                               └─ apply_identity_stitching
                                                    └─ YOUR STEP HERE
```

**Rules:**
- All prerequisites pass `True` for `use_cache_if_present`.
- Your own helper passes `False` (force recompute, invalidating the cache).
- Save result immediately after your helper call.

---

## Step 5 — Register the CLI handler

**File:** `src/session_summarizer/console/main.py`

### 5a — Add import (with other command imports at the top)
```python
from session_summarizer.commands.my_new_step import MyNewStepCommand
```

### 5b — Add command function (place after the command it follows in pipeline order)
```python
@app.command("my-new-step")
def my_new_step(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """One-sentence description of what this command does."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: MyNewStepCommand = MyNewStepCommand(session)
    command.execute(logger)
```

**Rules:**
- Handler only: collect args → `confirm_session` → `create_logger` → instantiate → `.execute(logger)`.
- No business logic in the handler.
- Place the function near other pipeline-ordered commands (typically after `apply-identity-stitching`, before `dump-human-format`).

---

## Step 6 — Add a launch profile to `.vscode/launch.json`

**File:** `.vscode/launch.json`

Add a new entry to the `configurations` array. Place it near other pipeline-stage entries (after the `-identity-stitch` entry is a good default).

**For a session command** (takes `--session`):
```json
{
    "name": "-my-new-step",
    "type": "debugpy",
    "request": "launch",
    "module": "session_summarizer",
    "console": "integratedTerminal",
    "args": [
        "my-new-step",
        "--session",
        "test",
    ]
}
```

**For a command with no session argument:**
```json
{
    "name": "-my-new-step",
    "type": "debugpy",
    "request": "launch",
    "module": "session_summarizer",
    "console": "integratedTerminal",
    "args": [
        "my-new-step",
    ]
}
```

**Naming rule:** Use `-<short-kebab>` — a leading dash followed by a short, readable abbreviation of the command name. Examples from the file: `apply-identity-stitching` → `-identity-stitch`, `add-embeddings` → `-embeddings`, `update-turn-end` → `-turn-end`. Prefer dropping common verbs (`apply-`, `compute-`) when the noun is unambiguous. Keep it short enough to scan in the VS Code launch dropdown.

---

## Step 7 — Add to ProcessPipelineCommand

**File:** `src/session_summarizer/commands/process_pipeline.py`

### 7a — Add import
```python
from .my_new_step import MyNewStepCommand
```

### 7b — Insert into `process_session` before the dump commands
```python
        StitichIdentitiesCommand(self.session_id).execute(self.logger)
        MyNewStepCommand(self.session_id).execute(self.logger)          # <-- add here
        DumpAndCompareTextsCommand(self.session_id).execute(self.logger)
        DumpHumanFormatCommand(self.session_id).execute(self.logger)
```

---

## Step 8 — Update dump utilities (only if output is a SpeechClipSet)

Skip this step if your command does not produce a `SpeechClipSet`.

### 8a — DumpHumanFormatCommand

**File:** `src/session_summarizer/commands/dump_human_format.py`

Add the new path to the `paths` list:

```python
        paths = [
            settings.base_diarized_path,
            settings.turn_end_updated_path,
            settings.first_stitched_path,
            settings.identified_speaker_path,
            settings.identity_stitched_path,
            settings.my_new_step_path,          # <-- add here
        ]
```

### 8b — DumpAndCompareTextsCommand

**File:** `src/session_summarizer/commands/dump_and_compare_texts.py`

1. Add import for your helper at the top.

2. After the `identity_stitching` block, add (using `True` — this command only reads, never invalidates):
```python
        new_clips: SpeechClipSet = apply_my_new_step(
            settings, session_dir, identity_stitched_clips, True, self, self.logger
        )
        new_clips.sort_clips()
        new_text = self.clean_and_dump_text(new_clips.plain_text(), session_dir / settings.my_new_step_path)
        new_eval = self.evaluate_texts(identity_stitched_text, new_text, "My New Step")
```

3. Add `new_eval` to the `results` list:
```python
        results: list[TranscriptionValidationResult] = [
            transcription_eval,
            align_eval,
            scored_eval,
            diarized_eval,
            merged_eval,
            identified_eval,
            identity_stitched_eval,
            new_eval,               # <-- add here
        ]
```

---

## Checklist

Before finishing, verify:

- [ ] `update-session-settings` run: new output `Path` field (and any tuning params) added to settings, YAML, and unit tests
- [ ] If significant 3rd-party/model work: domain code created in an appropriate domain folder, with no `SessionSettings` import
- [ ] Helper created in `src/session_summarizer/helpers/` with cache-check pattern; extracts plain values from settings before calling domain code
- [ ] Helper does NOT save to disk
- [ ] Command class created: all prerequisites use `True`, own helper uses `False`, result saved immediately
- [ ] CLI handler added to `main.py` — import + `@app.command` function with no business logic
- [ ] `.vscode/launch.json` entry added — `-<short-name>` profile with `"test"` session (or no session for non-session commands)
- [ ] `ProcessPipelineCommand` updated — import + call inserted before dump commands
- [ ] If SpeechClipSet output: `DumpHumanFormatCommand` paths list updated
- [ ] If SpeechClipSet output: `DumpAndCompareTextsCommand` updated with helper call (`True`), text dump, eval, and results list entry
- [ ] All tests pass: `uv run python -m pytest tests/ -v`
