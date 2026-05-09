from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.transcript_simplifier import simplify_transcript
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .save_session_clipset import SaveSessionClipsetCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class SimplifyTranscriptCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Simplify Transcript"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / f"{self.session_id}.json")
        self.outputs.append(session_dir / settings.paths.simplified_transcript)
        self.dependencies.append(SaveSessionClipsetCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / f"{self.session_id}.json")
        output_path: Path = session_dir / settings.paths.simplified_transcript
        output_path.parent.mkdir(parents=True, exist_ok=True)
        simplified_transcript: str = simplify_transcript(settings, session_dir, input_clips, self.logger, self.tracer)
        output_path.write_text(simplified_transcript, encoding="utf-8")
