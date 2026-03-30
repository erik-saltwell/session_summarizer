from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.commands.align_transcript import AlignTranscriptCommand
from session_summarizer.commands.clean_audio_command import CleanAudioCommand
from session_summarizer.commands.clean_session import CleanSessionCommand
from session_summarizer.commands.compute_vad_segments import ComputeVadSegmentsCommand
from session_summarizer.commands.score_confidence import ScoreConfidenceCommand
from session_summarizer.commands.transcribe_audio import TranscribeAudioCommand
from session_summarizer.commands.validate_transcribers import ValidateTranscribersCommand
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


@app.command("align-transcription")
def align_transcription(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: AlignTranscriptCommand = AlignTranscriptCommand(session)
    command.execute(logger)


@app.command("score-confidence")
def score_confidence(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ScoreConfidenceCommand = ScoreConfidenceCommand(session)
    command.execute(logger)


@app.command("clean-audio")
def clean_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: CleanAudioCommand = CleanAudioCommand(session)
    command.execute(logger)


@app.command("compute-vad-segments")
def compute_vad_segments(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to segment"),
) -> None:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ComputeVadSegmentsCommand = ComputeVadSegmentsCommand(session)
    command.execute(logger)


@app.command("validate-transcribers")
def validate_transcribers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Transcribe test audio with every registered transcriber and compare accuracy metrics."""
    confirm_session(session)
    logger: LoggingProtocol = create_logger()
    command: ValidateTranscribersCommand = ValidateTranscribersCommand(session)
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


# ---------------------------------------------------------------------------
# vad_segments_path  (optional — default: vad_segments.json)
# ---------------------------------------------------------------------------
# Where the VAD-based segment plan is written. This JSON file contains
# silence-aware cut points that downstream commands use to process long
# audio in chunks without splitting mid-speech.
# Relative paths are resolved from this file's directory.
# vad_segments_path: vad_segments.json


# ---------------------------------------------------------------------------
# min_segment_length / max_segment_length  (optional — defaults: 30 / 120)
# ---------------------------------------------------------------------------
# Bounds (in seconds) for each audio chunk produced by VAD segmentation.
#
#   min_segment_length — chunks shorter than this are merged with neighbours.
#       Lower values give finer granularity but increase overhead per chunk.
#       Default: 30
#
#   max_segment_length — no chunk will exceed this duration. If continuous
#       speech runs longer than this with no silence gap, a hard cut is made.
#       Must be large enough for your ASR model's context window.
#       Default: 120
#
# min_segment_length: 30
# max_segment_length: 120


# ---------------------------------------------------------------------------
# vad  (optional — VAD model hyperparameters)
# ---------------------------------------------------------------------------
# Controls the NeMo Voice Activity Detection model used to find speech and
# silence boundaries. The defaults work well for typical meeting recordings;
# tune these if you see too many false speech detections (lower onset) or
# missed speech (raise onset / lower offset).
#
#   model_name       — pretrained NeMo VAD model to use.
#                      Default: vad_multilingual_frame_marblenet
#
#   onset            — probability threshold to START a speech region (0.0–1.0).
#                      Higher = fewer false positives, may miss quiet speech.
#                      Default: 0.7
#
#   offset           — probability threshold to END a speech region (0.0–1.0).
#                      Lower = speech regions extend further into trailing silence.
#                      Must be ≤ onset for proper hysteresis.
#                      Default: 0.4
#
#   min_duration_on  — speech regions shorter than this (seconds) are discarded.
#                      Filters out clicks, coughs, and transient noise.
#                      Default: 0.3
#
#   min_duration_off — silence regions shorter than this (seconds) are bridged
#                      (treated as speech). Prevents choppy segmentation from
#                      brief pauses within sentences.
#                      Default: 0.3
#
#   pad_onset        — seconds of audio to include BEFORE each speech onset.
#                      Captures plosive consonants and breath that precede speech.
#                      Default: 0.1
#
#   pad_offset       — seconds of audio to include AFTER each speech offset.
#                      Captures word-final sounds and natural trailing silence.
#                      Default: 0.1
#
# vad:
#   model_name: vad_multilingual_frame_marblenet
#   onset: 0.7
#   offset: 0.4
#   min_duration_on: 0.3
#   min_duration_off: 0.3
#   pad_onset: 0.1
#   pad_offset: 0.1
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
