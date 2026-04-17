
Use these steps when creating a new Typer CLI command in this project.

A command consists of up to five layers:
1. **launch.json profile** (in .vscode/launch.json) - Every command gets a launch profile for easy debugging. profiles are ordered alphabetically by profile name.
2. **CLI handler** (in console/main.py) — Uses typer and handels validation, creates logger, instantiates command, executes it. Commands are stored alphabetically by function name.
3. **Command class** (in commands directory) — manages dependencies with other Commands/inputs/outputs, loads data, calls into helpers, saves results generates screen/file output.
4. **Helper function** — Reads settings, applies policy, performs internal work (without using major dependencies like ai models or similar) and calls into domain code.
5. **Domain code** (optional, lives in custom directory under src/session_summarizer) — the heavy implementation using 3rd-party packages or LLM models.

Most commands inherit from `SessionProcessingCommand` (they take a `--session` and run as a pipeline step). A smaller set are **standalone commands** that don't inherit from `SessionProcessingCommand` — see "Standalone commands" under Step 4 for that pattern.

---

## Conventions

| Thing | Convention | Example |
|---|---|---|
| launch.json profile | `kebab-case` with no leading dash | 'validate-diarization' |
| CLI command name | `kebab-case` | `my-new-step` |
| Command class | `PascalCaseCommand` | `MyNewStepCommand` |
| Helper function | `snake_case` | `apply_my_new_step` |
| Helper file | `src/.../helpers/<snake_case>.py` | `my_new_step.py` |
| Command file | `src/.../commands/<snake_case>.py` | `my_new_step.py` |
| Output path setting | `<snake_case>_path` | `my_new_step_path` |
| Domain module | `src/.../transcription/`, `diarization/`, etc. | see Step 2 |

---

## Step 1 — Add settings

Use the information in src/session_summarizer/settings/AGENTS.md to add a `Path` field for the new output file to `SessionSettings`. Also add any numeric/string tuning parameters the step needs.

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

The helper code does processing, but does not load prior pipeline steps from disk, or save final results to disk.

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
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    # Extract settings values; pass as plain arguments to domain code
    processor = MyProcessor(device=settings.device, threshold=settings.my_new_step_threshold)
    gpu_logger.report_gpu_usage("after loading model")

    result = processor.process(session_dir / settings.paths.cleaned_audio, logger)
    gpu_logger.report_gpu_usage("after inference")

    return result
```

**Rules:**
- Extract what domain code needs from settings here — domain code receives plain values, never `settings`.
- Return the result; do not save it.
- Helpers know nothing about caching — they always run and return a result. Caching is decided by the Command (see Step 4).
- Argument order is fixed by convention: `settings, session_dir, <prerequisite result(s)>, gpu_logger, logger`. All real helpers follow this order — match it so call sites stay consistent.
- `gpu_logger` is only meaningful for helpers that load models or run GPU inference. For pure-CPU transforms you can still accept it and simply not call it, or drop it from the signature entirely — see `helpers/text_punctuation.py` for a no-op example.

---

## Step 4 — Create the command class

**File:** `src/session_summarizer/commands/<snake_case_name>.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.text_punctuation import punctuate_text
from ..processing_results import SpeechClipSet
from ..protocols import SessionSettings
from .mark_backchannels import MarkBackchannelsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class PunctuateTextCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Punctuate Text"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.backchannel_marked)
        self.outputs.append(session_dir / settings.paths.punctuated_text)
        self.dependencies.append(MarkBackchannelsCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.backchannel_marked)
        # `self` is passed as the GpuLogger — SessionProcessingCommand implements the GpuLogger
        # protocol via its `report_gpu_usage` method.
        output_clips: SpeechClipSet = punctuate_text(settings, session_dir, input_clips, self, self.logger)
        self.save_speech_clip(output_clips, session_dir, settings.paths.punctuated_text)
```

**Rules:**
- Inherit from `SessionProcessingCommand` when the command takes a session id and runs as a pipeline step.
- Dependencies are appended with just `(self.session_id, self.tracer)`. `force` defaults to `False` on a dependency, so it re-runs only when `should_process` decides its outputs are stale (missing, or older than any declared input). **Do not pass `force=True` to dependencies** — that's reserved for the top-level CLI invocation (Step 5).
- `add_dependencies` populates three lists on `self`: `inputs` (files the helper reads), `outputs` (files that will be produced — used by the cache-staleness check in `should_process`), and `dependencies` (upstream commands).
- Inside `process_session`: load prior results from disk, call the helper, then save. Pass `self` as the `GpuLogger` argument and `self.logger` as the `LoggingProtocol`.
- Use `self.save_speech_clip(...)` / `self.save_alignment_result(...)` (inherited from the base class) to persist results — do not invoke `.save_to_json` directly inside the command.
- Either `from ..protocols import SessionSettings` or `from ..settings import SessionSettings` is fine; both are used interchangeably throughout the codebase.

### Commands with extra options

A command may declare additional `@dataclass` fields (with defaults, so they don't conflict with the inherited `session_id` / `tracer` / `force` fields that have no default). The CLI handler passes them as keyword args after `session, tracer`. Example — `CreateSpeakerClipsCommand` adds `temp_folder: Path` and `use_multi_speaker_clips: bool`, and its handler instantiates it as:

```python
CreateSpeakerClipsCommand(session, tracer, use_multi_speaker_clips=False, temp_folder=Path(temp_folder))
```

### Standalone commands — no session, no pipeline

When the command does not process a session and has no upstream prerequisites (e.g. clearing logs, registering speakers from disk), **skip `SessionProcessingCommand`** entirely and build a plain dataclass. No `session_id`, no `tracer`, no `add_dependencies`, no caching, no `process_session`.

```python
from __future__ import annotations

from dataclasses import dataclass

from ..protocols import LoggingProtocol, NullLogger


@dataclass
class ClearLogsCommand:
    """Short one-line description of what this does."""

    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Clear Logs"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        # ...do the work directly...
```

The CLI handler for a standalone command is correspondingly minimal — no `confirm_session`, no `_set_seed`:

```python
@app.command("clear-logs")
def clear_logs() -> None:
    logger, _tracer = initialize_logging()
    ClearLogsCommand().execute(logger)
```

See `commands/clear_logs.py` and `commands/register_speakers.py` for the two existing examples.

---

## Step 5 — Register the CLI handler

**File:** `src/session_summarizer/console/main.py`

### 5a — Add import (with other command imports at the top)
```python
from session_summarizer.commands.my_new_step import MyNewStepCommand
```

### 5b — Add command function (place after the command it follows in pipeline order)
```python
@app.command("punctuate-text")
def punctuate_text(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: PunctuateTextCommand = PunctuateTextCommand(session, tracer, force=True)
    command.execute(logger)
```

**Rules:**
- Handler only: collect args → `confirm_session` → `create_logger` → instantiate → `.execute(logger)`.
- No business logic in the handler.
- Place the function in alphabetical order by function name

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

**Naming rule:** Use '<short-kebab>` — a short, readable abbreviation of the command name. Examples from the file: `apply-identity-stitching` → `identity-stitch`, `add-embeddings` → `embeddings`, `update-turn-end` → `turn-end`. Prefer dropping common verbs (`apply-`, `compute-`) when the noun is unambiguous. Keep it short enough to scan in the VS Code launch dropdown.
**Ordering:** Profiles are placed in alphabetical order by command name.

---

## Checklist

Before finishing, verify:

- [ ] `update-session-settings` run: new output `Path` field (and any tuning params) added to settings, YAML, and unit tests
- [ ] If significant 3rd-party/model work: domain code created in an appropriate domain folder, with no `SessionSettings` import
- [ ] Helper created in `src/session_summarizer/helpers/` — signature follows `(settings, session_dir, <prerequisite>, gpu_logger, logger)`; extracts plain values from settings before calling domain code; calls `gpu_logger.report_gpu_usage(...)` around GPU-heavy steps
- [ ] Helper does NOT save to disk and does NOT consult any cache flag
- [ ] Command class created: dependencies appended with `(self.session_id, self.tracer)` only — no `force=True` on dependencies; helper called with `self` as `GpuLogger` and `self.logger` as `LoggingProtocol`; result saved via inherited `self.save_*` method
- [ ] For a standalone (non-session) command: plain `@dataclass` with only a `logger` field, no `SessionProcessingCommand` base, no `add_dependencies` / `process_session`, CLI handler skips `confirm_session` / `_set_seed`
- [ ] CLI handler added to `main.py` — import + `@app.command` function with no business logic; top-level command instantiated with `force=True`
- [ ] `.vscode/launch.json` entry added — `<short-name>` profile with `"test"` session (or no session for non-session commands)
