from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.dangling_sentence_fixer import fix_dangling_sentences
from ..processing_results import SpeechClipSet
from ..protocols import (
    SessionSettings,
)
from .diarizationlm_command import DiarizationLMCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class DanglingSentenceFixCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Fix Dangling Sentences"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.diarizationlm_processed)
        self.outputs.append(session_dir / settings.paths.dangling_sentence_fix)
        self.dependencies.append(DiarizationLMCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.diarizationlm_processed)
        output_clips: SpeechClipSet = fix_dangling_sentences(
            settings, session_dir, input_clips, self, self.logger, self.tracer
        )
        self.save_speech_clip(output_clips, session_dir, settings.paths.dangling_sentence_fix)
