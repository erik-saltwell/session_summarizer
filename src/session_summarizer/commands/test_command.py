from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class TestCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Test"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        return

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        pass
