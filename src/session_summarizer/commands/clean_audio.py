from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_cleaner import clean_audio
from ..settings.session_settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class CleanAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Clean Audio"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.audio_file)
        self.outputs.append(session_dir / settings.cleaned_audio_file)

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clean_audio(settings, session_dir, self, self.logger)
