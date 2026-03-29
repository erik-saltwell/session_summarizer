from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from session_summarizer.utils import common_paths

from ..logging import CompositeLogger, FileLogger, RichConsoleLogger
from ..protocols import CommmandProtocol, LoggingProtocol


@dataclass
class CommandRunner:
    session_id: str

    def create_logger(self) -> LoggingProtocol:
        console = Console()
        console_logger: RichConsoleLogger = RichConsoleLogger(console)
        logfile_path = common_paths.generate_logfile_path()
        file_logger: FileLogger = FileLogger(logfile_path, verbose_training=True)
        return CompositeLogger([console_logger, file_logger])

    def Run(self, command: CommmandProtocol) -> None:
        # logger: LoggingProtocol = self.create_logger()
        # settings: SessionSettings = SessionSettings.load_cascading(self.session_id)
        return
