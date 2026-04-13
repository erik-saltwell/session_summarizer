from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.diarizationlm_refiner import apply_diarizationlm
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .diarize_audio import DiarizeAudioCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class DiarizationLMCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "DiarizationLM"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.base_diarization)
        self.outputs.append(session_dir / settings.paths.diarizationlm_processed)
        self.dependencies.append(DiarizeAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.base_diarization)
        refined_clips: SpeechClipSet = apply_diarizationlm(settings, session_dir, clips, self, self.logger)

        self.save_speech_clip(refined_clips, session_dir, settings.paths.diarizationlm_processed)
