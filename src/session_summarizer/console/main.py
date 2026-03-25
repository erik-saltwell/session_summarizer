from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.protocols import command_protocol, transcriber_protocol
from session_summarizer.utils import common_paths

from ..alignment import ParakeetCTCAligner
from ..commands.align_audio import AlignAudioCommand
from ..commands.clean_original_audio import CleanOriginalAudioCommand
from ..commands.diarize_audio import DiarizeAudioCommand
from ..commands.register_speaker import RegisterSpeakerCommand
from ..commands.transcribe_audio import TranscribeAudioCommand
from ..diarization import MsddDiarizer
from ..protocols import CompositeLogger, LoggingProtocol
from ..speaker import ERes2NetV2Embedder
from ..transcription import CanaryQwenTranscriber, WhisperLargeTranscriber
from ..utils import flush_gpu_memory
from ..utils.logging_config import configure_logging
from .console_validation import _validate_directory_name
from .file_logging_protocol import FileLogger
from .rich_logging_protocol import RichConsoleLogger

load_dotenv()
configure_logging()

flush_gpu_memory()

app = typer.Typer(
    name="session-summarizer",
    add_completion=True,
    help="CLI for session-summarizer",
)

LOG_FILENAME: str = "session_summarizer.log"


def create_logger() -> LoggingProtocol:
    console = Console()
    console_logger: RichConsoleLogger = RichConsoleLogger(console)
    file_logger: FileLogger = FileLogger(LOG_FILENAME, verbose_training=True)
    return CompositeLogger([console_logger, file_logger])


def seconds_since(start: datetime) -> float:
    return (datetime.now() - start).total_seconds()


@app.command("clean-audio")
def clean_audio(session: str = typer.Option(..., "--session", "-s", help="ID of the session to process")) -> None:
    """Simple smoke command."""
    _validate_directory_name(str(common_paths.session_path(session)))
    common_paths.ensure_directory(common_paths.session_path(session))
    logger: LoggingProtocol = create_logger()

    command: command_protocol.CommmandProtocol = CleanOriginalAudioCommand(session)
    command.execute(logger)


_ENGINE_CHOICES = ["whisper", "canary"]


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
    engine: str = typer.Option("whisper", "--engine", "-e", help=f"Transcription engine: {', '.join(_ENGINE_CHOICES)}"),
    model_size: str = typer.Option("large", "--model-size", help="Whisper/WhisperX model size (e.g. large, large-v2)"),
    device: str = typer.Option("cuda", "--device", help="Torch device (cuda or cpu)"),
) -> None:
    """Transcribe normalized_audio.wav for a session, writing transcript.json."""
    if engine not in _ENGINE_CHOICES:
        raise typer.BadParameter(f"--engine must be one of: {', '.join(_ENGINE_CHOICES)}")

    _validate_directory_name(str(common_paths.session_path(session)))
    logger: LoggingProtocol = create_logger()

    transcriber: transcriber_protocol.TranscriberProtocol
    if engine == "canary":
        transcriber = CanaryQwenTranscriber(device=device)
    else:
        transcriber = WhisperLargeTranscriber(model_size=model_size, device=device)

    TranscribeAudioCommand(session_id=session, transcriber=transcriber).execute(logger)


@app.command("align-words")
def align_words(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to align"),
    device: str = typer.Option("cuda", "--device", help="Torch device (cuda or cpu)"),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size for CTC alignment"),
) -> None:
    """Align words from transcript.json against normalized_audio.wav, writing word_alignments.json."""
    _validate_directory_name(str(common_paths.session_path(session)))
    logger: LoggingProtocol = create_logger()

    aligner = ParakeetCTCAligner(device=device, batch_size=batch_size)
    AlignAudioCommand(session_id=session, aligner=aligner).execute(logger)


@app.command("diarize")
def diarize(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to diarize"),
    number_of_speakers: int = typer.Option(..., "--number-of-speakers", help="Exact number of speakers in the audio"),
    device: str = typer.Option("cuda", "--device", help="Torch device (cuda or cpu)"),
) -> None:
    """Diarize normalized_audio.wav using MSDD, writing diarization.json."""
    _validate_directory_name(str(common_paths.session_path(session)))
    logger: LoggingProtocol = create_logger()

    diarizer = MsddDiarizer(device=device, num_speakers=number_of_speakers)
    DiarizeAudioCommand(session_id=session, diarizer=diarizer).execute(logger)


@app.command("register-speaker")
def register_speaker(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session"),
    speaker_name: str = typer.Option(..., "--speaker-name", help="Name to register for this speaker"),
    wav_file: Path = typer.Option(..., "--wav-file", help="Path to a WAV file containing this speaker's voice"),  # noqa: B008
    device: str = typer.Option("cuda", "--device", help="Torch device (cuda or cpu)"),
) -> None:
    """Extract an ERes2NetV2 embedding for a speaker and save to registered_speakers.yaml."""
    _validate_directory_name(str(common_paths.session_path(session)))
    common_paths.ensure_directory(common_paths.session_path(session))
    logger: LoggingProtocol = create_logger()

    embedder = ERes2NetV2Embedder(device=device)
    RegisterSpeakerCommand(
        session_id=session,
        speaker_name=speaker_name,
        wav_file=wav_file,
        embedder=embedder,
    ).execute(logger)


@app.command("test")
def test() -> None:
    """Simple smoke command."""
    console = Console()
    console.print("[green]Hello from test[/green]")


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
