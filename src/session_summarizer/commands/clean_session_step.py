from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from session_summarizer.settings.session_settings import SessionSettings

from .session_processing_command import SessionProcessingCommand


@dataclass
class CleanSessionStepCommand(SessionProcessingCommand):
    commands_to_clean: list[SessionProcessingCommand] = field(
        default_factory=list
    )  # outputs from any command added here will be deleted.

    def name(self) -> str:
        return "Clean Session Step"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        return

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        for command in self.commands_to_clean:
            for output in command.outputs:
                output.unlink(missing_ok=True)
