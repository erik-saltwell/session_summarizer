from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.infer_speakers import InferSpeakersResult, infer_speakers, update_session_inferred_speaker_settings
from ..processing_results import SpeechClipSet
from ..protocols import SessionSettings
from .dangling_sentece_fix import DanglingSentenceFixCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class InferSpeakersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Infer Speakers"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.dangling_sentence_fix)
        self.outputs.append(session_dir / settings.paths.inferred_speakers)
        self.dependencies.append(DanglingSentenceFixCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.dangling_sentence_fix)
        result: InferSpeakersResult = infer_speakers(settings, session_dir, input_clips, self.logger, self.tracer)
        self.save_speech_clip(result.clips, session_dir, settings.paths.inferred_speakers)
        update_session_inferred_speaker_settings(session_dir / "settings.yaml", result.participants)
