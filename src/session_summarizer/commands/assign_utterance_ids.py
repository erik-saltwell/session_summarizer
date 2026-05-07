from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.assign_utterance_ids import assign_utterance_ids
from ..processing_results import SpeechClipSet
from ..protocols import SessionSettings
from .punctuate_text import PunctuateTextCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class AssignUtteranceIdsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Assign Utterance Ids"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.punctuated_text)
        self.outputs.append(session_dir / settings.paths.utterance_ids_annotated)
        self.dependencies.append(PunctuateTextCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.punctuated_text)
        output_clips: SpeechClipSet = assign_utterance_ids(settings, session_dir, input_clips, self, self.logger)
        self.save_speech_clip(output_clips, session_dir, settings.paths.utterance_ids_annotated)
