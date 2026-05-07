from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..protocols import CommmandProtocol, LoggingProtocol
from ..utils import Tracer, common_paths


class CommandRunnerHost(Protocol):
    def should_run_command_agianst_session(self, session_id: str) -> bool: ...
    def get_command(self, session_id: str, logger: LoggingProtocol, tracer: Tracer) -> CommmandProtocol: ...


def process_sessions(host: CommandRunnerHost, logger: LoggingProtocol, tracer: Tracer) -> None:
    experiments_dir: Path = common_paths.data_dir()
    for child in experiments_dir.iterdir():
        if not child.is_dir() or not host.should_run_command_agianst_session(child.name):
            continue
        logger.report_message(f"Processing Session: {child.name}")
        command: CommmandProtocol = host.get_command(child.name, logger, tracer)
        command.execute(logger)
