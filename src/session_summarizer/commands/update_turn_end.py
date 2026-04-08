from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.update_turn_end import update_turn_end
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .diarizationlm_command import DiarizationLMCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class UpdateTurnEndCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Update Turn End"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.diarizationlm_processed_path)
        self.outputs.append(session_dir / settings.turn_end_updated_path)
        self.dependencies.append(DiarizationLMCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        diarized_clips: SpeechClipSet = SpeechClipSet.load_from_json(
            session_dir / settings.diarizationlm_processed_path
        )

        turn_clips: SpeechClipSet = update_turn_end(settings, session_dir, diarized_clips, self, self.logger)
        self.save_speech_clip(turn_clips, session_dir, settings.turn_end_updated_path)
