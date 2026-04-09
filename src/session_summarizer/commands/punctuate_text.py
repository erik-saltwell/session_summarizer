from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.text_punctuation import punctuate_text
from ..processing_results import SpeechClipSet
from ..protocols import (
    SessionSettings,
)
from .session_processing_command import SessionProcessingCommand
from .stitch_identities import StitichIdentitiesCommand


@dataclass
class PunctuateTextCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Punctuate Text"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.identity_stitched_path)
        self.outputs.append(session_dir / settings.punctuated_text_path)
        self.dependencies.append(StitichIdentitiesCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)
        output_clips: SpeechClipSet = punctuate_text(settings, session_dir, input_clips, self, self.logger)
        self.save_speech_clip(output_clips, session_dir, settings.punctuated_text_path)
