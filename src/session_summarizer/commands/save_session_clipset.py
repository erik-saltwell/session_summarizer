from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .assign_utterance_ids import AssignUtteranceIdsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class SaveSessionClipsetCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Save Session Clipset"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.utterance_ids_annotated)
        self.outputs.append(session_dir / f"{self.session_id}.json")
        self.dependencies.append(AssignUtteranceIdsCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.utterance_ids_annotated)
        self.save_speech_clip(clips, session_dir, Path(f"{self.session_id}.json"))
