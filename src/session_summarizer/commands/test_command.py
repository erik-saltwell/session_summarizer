from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.command_runner import CommandRunnerHost, process_sessions
from ..protocols import CommmandProtocol, LoggingProtocol
from ..settings.session_settings import SessionSettings
from ..utils import Tracer
from .assign_utterance_ids import AssignUtteranceIdsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class TestCommand(SessionProcessingCommand, CommandRunnerHost):
    def should_run_command_agianst_session(self, session_id: str) -> bool:
        return session_id.startswith("Delta") or session_id.startswith("2026")

    def get_command(self, session_id: str, logger: LoggingProtocol, tracer: Tracer) -> CommmandProtocol:
        return AssignUtteranceIdsCommand(session_id, tracer, False, logger)

    def name(self) -> str:
        return "Test"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        return

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        process_sessions(self, self.logger, self.tracer)
