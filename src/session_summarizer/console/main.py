from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from importlib.metadata import version as dist_version

import typer
from dotenv import load_dotenv
from rich.console import Console

from ..protocols import CompositeLogger, LoggingProtocol
from ..utils.logging_config import configure_logging
from .file_logging_protocol import FileLogger
from .rich_logging_protocol import RichConsoleLogger

load_dotenv()
configure_logging()


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
