from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings

from ..helpers.audio_cleaner import clean_audio
from .session_processing_command import SessionProcessingCommand


@dataclass
class CleanAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Clean Audio"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clean_audio(settings, session_dir, False, self, self.logger)
