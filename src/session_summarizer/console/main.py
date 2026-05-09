from __future__ import annotations

import os

# Suppress TensorFlow C++ and oneDNN log spam before any ML library is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
import sys
from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version
from pathlib import Path

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.commands.punctuate_text import PunctuateTextCommand
from session_summarizer.utils import common_paths

from ..commands.add_embeddings import AddEmbeddingsCommand
from ..commands.assign_utterance_ids import AssignUtteranceIdsCommand
from ..commands.clean_audio import CleanAudioCommand
from ..commands.clean_session import CleanSessionCommand
from ..commands.clean_session_step import CleanSessionStepCommand
from ..commands.clear_logs import ClearLogsCommand
from ..commands.create_speaker_clips import (
    CreateKnownSpeakerClipsCommand,
    CreateSpeakerClipsFromInferredSpeakersCommand,
)
from ..commands.diarize_audio import DiarizeAudioCommand
from ..commands.document_dependencies import DocumentDependenciesCommand
from ..commands.identify_speakers import IdentifySpeakersCommand
from ..commands.infer_speakers import InferSpeakersCommand
from ..commands.mark_backchannels import MarkBackchannelsCommand
from ..commands.merge_speaker_clips import MergeSpeakerClipsCommand
from ..commands.register_speakers import RegisterSpeakersCommand
from ..commands.remove_outlier_speaker_clips import RemoveOutlierSpeakerClipsCommand
from ..commands.save_session_clipset import SaveSessionClipsetCommand
from ..commands.simplify_transcript import SimplifyTranscriptCommand
from ..commands.stitch_identities import StitichIdentitiesCommand
from ..commands.summarize_session import SummarizeSessionCommand
from ..commands.test_command import TestCommand
from ..commands.validate_diarization import ValidateDiarizationCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import Tracer, configure_logging, flush_gpu_memory, initialize_request, initialize_tracing
from .console_validation import _validate_directory_exists
from .generate_settings import get_sample_settings

load_dotenv()
configure_logging()
initialize_tracing()

# Set random seeds for reproducible model inference


def _set_seed(session_id: str) -> None:
    settings: SessionSettings = SessionSettings.load_cascading(session_id)
    seed: int = settings.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


flush_gpu_memory()

app = typer.Typer(
    name="session-summarizer",
    add_completion=True,
    help="CLI for session-summarizer",
)


def initialize_logging() -> tuple[LoggingProtocol, Tracer]:
    logger: LoggingProtocol
    tracer: Tracer

    request_id = initialize_request()
    tracer = Tracer()

    console = Console(file=sys.__stdout__)
    # error_console = Console(file=sys.__stderr__)
    console_logger: RichConsoleLogger = RichConsoleLogger(console)

    logfile_path = common_paths.generate_logfile_path()
    file_logger: FileLogger = FileLogger(logfile_path, verbose_training=True)
    logger = CompositeLogger([console_logger, file_logger])

    # logger = console_logger
    logger.report_message(f"[blue]Session id: {request_id}[/blue]")
    return logger, tracer


def confirm_session(session_id: str) -> None:
    session_dir = common_paths.session_dir(session_id)
    errors: list[str] = _validate_directory_exists(session_dir)
    if errors and len(errors) > 0:
        console: Console = Console()
        for error in errors:
            console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)


@app.command("add-embeddings")
def add_embeddings(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate speaker embeddings for each speech clip and save to disk."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: AddEmbeddingsCommand = AddEmbeddingsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("apply-identity-stitching")
def apply_identity_stitiching(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Score each speech clip with end-of-turn probability and set the END_OF_TURN flag."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: StitichIdentitiesCommand = StitichIdentitiesCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("assign-utterance-ids")
def assign_utterance_ids(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Stamp each speech clip with a stable <campaign_id>_<session_id>_<n> utterance id."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: AssignUtteranceIdsCommand = AssignUtteranceIdsCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("save-session-clipset")
def save_session_clipset(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Save the utterance-id-stamped clipset under <session_id>.json."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: SaveSessionClipsetCommand = SaveSessionClipsetCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("create-known-speaker-clips")
def create_known_speaker_clips(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
    temp_folder: str = typer.Option(
        ..., "--temp-folder", "-t", help="Name of temp folder inside voice samples to hold output"
    ),
) -> None:
    """Save known identified-speaker clips as individual audio files."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CreateKnownSpeakerClipsCommand = CreateKnownSpeakerClipsCommand(
        session, tracer, use_multi_speaker_clips=False, temp_folder=Path(temp_folder)
    )
    command.execute(logger)


@app.command("create-speaker-clips-from-inferred-speakers")
def create_speaker_clips_from_inferred_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Save inferred-speaker clips into top-level voice sample folders."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CreateSpeakerClipsFromInferredSpeakersCommand = CreateSpeakerClipsFromInferredSpeakersCommand(
        session, tracer, force=True
    )
    command.execute(logger)


@app.command("clean-audio")
def clean_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: CleanAudioCommand = CleanAudioCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("clean-diarization")
def clean_diarization(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CleanSessionStepCommand = CleanSessionStepCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("clean-session")
def clean_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to clean"),
) -> None:
    """Delete all generated files in a session folder, keeping settings.yaml and the original audio."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: CleanSessionCommand = CleanSessionCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("clear-logs")
def clear_logs() -> None:
    """Delete all files in the logs directory."""
    logger, tracer = initialize_logging()
    ClearLogsCommand().execute(logger)


@app.command("merge-speaker-clips")
def merge_speaker_clips(
    speaker: str = typer.Option(
        ..., "--speaker", "-s", help="Speaker label — must match a subdirectory in voice_samples/"
    ),
    output_folder: str = typer.Option(..., "--output-folder", "-o", help="Folder to write the merged clips into"),
) -> None:
    """Merge short clips for a speaker until all are >= speaker_clips.min_duration_seconds."""
    logger, tracer = initialize_logging()
    command: MergeSpeakerClipsCommand = MergeSpeakerClipsCommand(
        speaker_label=speaker,
        output_folder=Path(output_folder),
    )
    command.execute(logger)


@app.command("remove-outlier-speaker-clips")
def remove_outlier_speaker_clips(
    speaker: str = typer.Option(
        ..., "--speaker", "-s", help="Speaker label — must match a subdirectory in voice_samples/"
    ),
    output_folder: str = typer.Option(..., "--output-folder", "-o", help="Folder to write the merged clips into"),
) -> None:
    """Merge short clips for a speaker until all are >= speaker_clips.min_duration_seconds."""
    logger, tracer = initialize_logging()
    command: RemoveOutlierSpeakerClipsCommand = RemoveOutlierSpeakerClipsCommand(
        speaker_label=speaker,
        output_folder=Path(output_folder),
    )
    command.execute(logger)


@app.command("diarize-audio")
def diarize_audio(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to diarize"),
) -> None:
    """Diarize session audio using ElevenLabs speaker diarization."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()
    command: DiarizeAudioCommand = DiarizeAudioCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("document-dependencies")
def document_dependencies_cmd() -> None:
    """Inspect all pipeline commands and write a Mermaid file/command dependency graph."""
    logger, _tracer = initialize_logging()
    command: DocumentDependenciesCommand = DocumentDependenciesCommand()
    command.execute(logger)


@app.command("generate-sample-settings")
def generate_sample_settings() -> None:
    """Generate a well-documented sample settings.yaml in the data directory."""
    console = Console()
    target = common_paths.data_dir() / "settings.yaml"

    common_paths.ensure_directory(common_paths.data_dir())
    target.write_text(get_sample_settings(), encoding="utf-8")
    console.print(f"[green]Sample settings written to {target}[/green]")
    console.print("[dim]Edit the file to match your session before running other commands.[/dim]")


@app.command("identify-speakers")
def identify_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Identify speakers in each speech clip by comparing embeddings to registered attendees."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: IdentifySpeakersCommand = IdentifySpeakersCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("infer-speakers")
def infer_speakers(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Infer role-based speaker identities from transcript text."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: InferSpeakersCommand = InferSpeakersCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("mark-backchannels")
def mark_backchannels(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Mark short acknowledgement clips as backchannels."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: MarkBackchannelsCommand = MarkBackchannelsCommand(session, tracer, force=True)
    command.execute(logger)


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


@app.command("register-speakers")
def register_speakers() -> None:
    """Merge clips, remove outliers, and register centroid embeddings into registered_speakers.yaml."""
    logger, tracer = initialize_logging()
    RegisterSpeakersCommand().execute(logger)


@app.command("simplify-transcript")
def simplify_transcript(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to process"),
) -> None:
    """Generate a cleaned narrative transcript from punctuated speech clips."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: SimplifyTranscriptCommand = SimplifyTranscriptCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("summarize-session")
def summarize_session(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: SummarizeSessionCommand = SummarizeSessionCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("test")
def test(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
) -> None:
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: TestCommand = TestCommand(session, tracer, force=True)
    command.execute(logger)


@app.command("validate-diarization")
def validate_diarization(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to use for validation"),
) -> None:
    """Evaluate diarization quality across pipeline stages and display a metrics comparison table."""
    confirm_session(session)
    _set_seed(session)
    logger, tracer = initialize_logging()

    command: ValidateDiarizationCommand = ValidateDiarizationCommand(session, tracer, force=True)
    command.execute(logger)


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
