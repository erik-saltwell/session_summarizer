from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths

from ..protocols import LoggingProtocol, NullLogger


@dataclass
class ClearLogsCommand:
    """Delete all files in the logs directory."""

    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Clear Logs"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        logs_dir = common_paths.logs_dir()

        if not logs_dir.exists():
            logger.report_message(f"[yellow]Logs directory does not exist: {logs_dir}[/yellow]")
            return

        files = [f for f in logs_dir.iterdir() if f.is_file()]
        if not files:
            logger.report_message(f"[yellow]No log files to delete in {logs_dir}[/yellow]")
            return

        for file in files:
            file.unlink()

        logger.report_message(f"[green]Deleted {len(files)} log file(s) from {logs_dir}[/green]")
