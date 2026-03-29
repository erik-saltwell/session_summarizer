from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.commands.align_transcript import AlignTranscriptCommand
from session_summarizer.commands.clean_audio_command import CleanAudioCommand
from session_summarizer.commands.clean_session import CleanSessionCommand
from session_summarizer.commands.transcribe_audio import TranscribeAudioCommand
from session_summarizer.utils import common_paths

from ..commands.register_speakers import RegisterSpeakersCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..utils import flush_gpu_memory
from ..utils.logging_config import configure_logging
from .console_validation import _validate_directory_exists

load_dotenv()
configure_logging()

flush_gpu_memory()

app = typer.Typer(
    name="session-summarizer",
    add_completion=True,
    help="CLI for session-summarizer",
)


def create_logger() -> LoggingProtocol:
    console = Console()
    console_logger: RichConsoleLogger = RichConsoleLogger(console)
    logfile_path = common_paths.generate_logfile_path()
    file_logger: FileLogger = FileLogger(logfile_path, verbose_training=True)
    return CompositeLogger([console_logger, file_logger])


def confirm_session(session_id: str) -> None:
    session_dir = common_paths.session_dir(session_id)
    errors: list[str] = _validate_directory_exists(session_dir)
    if errors and len(errors) > 0:
        console: Console = Console()
        for error in errors:
            console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: TranscribeAudioCommand = TranscribeAudioCommand(session)
    command.execute(logger)


@app.command("align")
def align(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: AlignTranscriptCommand = AlignTranscriptCommand(session)
    command.execute(logger)


@app.command("clean")
def clean(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: CleanAudioCommand = CleanAudioCommand(session)
    command.execute(logger)


@app.command("clean-session")
def clean_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: CleanSessionCommand = CleanSessionCommand(session)
    command.execute(logger)


_SAMPLE_SETTINGS = """\
# ============================================================================
# Session Summarizer — settings.yaml
# ============================================================================
#
# This file configures a session-summarizer run. It can live in two places:
#
#   1. data/settings.yaml           — shared defaults for every session
#   2. data/<session-id>/settings.yaml — per-session overrides
#
# When both exist, per-session values override the shared defaults.
# ============================================================================


# ---------------------------------------------------------------------------
# attendees  (REQUIRED)
# ---------------------------------------------------------------------------
# A list of speaker names expected in the session. This drives diarization
# (the number of speakers the model should look for) and labels in the final
# transcript. Every entry must be a non-empty string.
#
# Example:
#   attendees:
#     - Alice
#     - Bob
#     - Charlie
attendees:
  - Speaker1
  - Speaker2


# ---------------------------------------------------------------------------
# audio_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Path to the original recording. Supported formats:
#   .m4a  .mp3  .wav  .flac  .ogg  .opus  .wma  .aac  .webm
#
# Relative paths are resolved from the directory that contains this file.
# Absolute paths are used as-is (and must point to an existing file).
#
# Example:
#   audio_file: meeting_2025-03-29.m4a
audio_file: original.m4a


# ---------------------------------------------------------------------------
# cleaned_audio_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the noise-reduced audio is written (or read from, if it already
# exists). Relative paths are resolved from this file's directory.
cleaned_audio_file: cleaned_audio.wav


# ---------------------------------------------------------------------------
# transcript_file  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the transcript JSON is written (or read from, if it already exists).
# Relative paths are resolved from this file's directory.
transcript_file: transcript.json


# ---------------------------------------------------------------------------
# aligned_transcript_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the word-aligned transcript is written (or read from, if it already
# exists). Word alignment maps each word to a precise start/end timestamp
# using CTC forced alignment — more accurate than segment-level timing.
# Relative paths are resolved from this file's directory.
aligned_transcript_path: aligned_transcript.json


# ---------------------------------------------------------------------------
# confidence_transcript_path  (REQUIRED)
# ---------------------------------------------------------------------------
# Where the transcript annotated with per-word confidence scores is written
# (or read from, if it already exists). Confidence scores (0.0–1.0) indicate
# how certain the model was about each word; useful for post-processing,
# review prioritisation, and filtering low-confidence segments.
# Relative paths are resolved from this file's directory.
confidence_transcript_path: confidence_transcript.json


# ---------------------------------------------------------------------------
# device  (REQUIRED)
# ---------------------------------------------------------------------------
# Compute device for model inference. Allowed values:
#   cuda  — use the GPU (requires a CUDA-capable NVIDIA GPU)
#   cpu   — use the CPU (much slower, but works everywhere)
device: cuda
"""


@app.command("generate-sample-settings")
def generate_sample_settings() -> None:
    """Generate a well-documented sample settings.yaml in the data directory."""
    console = Console()
    target = common_paths.data_dir() / "settings.yaml"

    if target.exists():
        console.print(f"[red]Settings file already exists: {target}[/red]")
        console.print("[dim]Remove or rename it first if you want a fresh sample.[/dim]")
        raise typer.Exit(1)

    common_paths.ensure_directory(common_paths.data_dir())
    target.write_text(_SAMPLE_SETTINGS, encoding="utf-8")
    console.print(f"[green]Sample settings written to {target}[/green]")
    console.print("[dim]Edit the file to match your session before running other commands.[/dim]")


@app.command("register-speakers")
def register_speakers() -> None:
    """Register all speakers from voice_samples directory into registered_speakers.yaml."""
    logger: LoggingProtocol = create_logger()
    RegisterSpeakersCommand().execute(logger)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if not value:
        return

    # IMPORTANT: distribution name (pyproject.toml [project].name), often hyphenated.
    # Example: "my-tool" even if your import package is "my_tool".
    DIST_NAME = "session-summarizer"

    console = Console()

    try:
        pkg_version = dist_version(DIST_NAME)
        md = metadata(DIST_NAME)
        try:
            pkg_name = md["Name"]
        except KeyError:
            pkg_name = DIST_NAME

        console.print(f"{pkg_name} {pkg_version}")
    except PackageNotFoundError:
        # Running from source without an installed distribution
        console.print(f"{DIST_NAME} 0.0.0+unknown")

    raise typer.Exit()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root command group for reddit_rpg_miner."""
    # Intentionally empty: this forces Typer to keep subcommands like `test`.
    pass


if __name__ == "__main__":
    app()
