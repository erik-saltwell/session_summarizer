from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import typer
from dotenv import load_dotenv
from rich.console import Console

from session_summarizer.protocols import transcriber_protocol
from session_summarizer.utils import common_paths

from ..commands.register_speakers import RegisterSpeakersCommand
from ..commands.transcribe_audio import TranscribeAudioCommand
from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import LoggingProtocol
from ..transcription import CanaryQwenTranscriber, ParakeetCTCAligner
from ..utils import flush_gpu_memory
from ..utils.logging_config import configure_logging
from .console_validation import _validate_directory_name

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


@app.command("transcribe")
def transcribe(
    session: str = typer.Option(..., "--session", "-s", help="ID of the session to transcribe"),
    device: str = typer.Option("cuda", "--device", help="Torch device (cuda or cpu)"),
) -> None:
    """Clean audio and transcribe a session, writing transcript.json."""

    _validate_directory_name(str(common_paths.session_dir(session)))
    logger: LoggingProtocol = create_logger()

    transcriber: transcriber_protocol.TranscriberProtocol = CanaryQwenTranscriber(device=device)
    aligner = ParakeetCTCAligner(device=device)

    TranscribeAudioCommand(session_id=session, transcriber=transcriber, aligner=aligner).execute(logger)


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
