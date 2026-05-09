from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.audio_cleaner import clean_audio_eleven_labs
from ..settings import SessionSettings
from ..utils import common_paths
from .session_processing_command import SessionProcessingCommand


@dataclass
class CleanAudioElevenLabsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Clean Audio (ElevenLabs)"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.source_audio)
        self.outputs.append(session_dir / settings.paths.cleaned_audio)

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clean_audio_eleven_labs(settings, session_dir, True, self, self.logger, self.tracer)
