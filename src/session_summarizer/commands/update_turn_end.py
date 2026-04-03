from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet

from ..helpers.update_turn_end import update_turn_end
from ..settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class UpdateTurnEndCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Update Turn End"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clips: SpeechClipSet = update_turn_end(settings, session_dir, False, self, self.logger)
        clips.save_to_json(session_dir / settings.base_diarized_path)
