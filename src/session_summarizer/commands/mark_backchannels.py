from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.backchannel_marker import mark_backchannels
from ..processing_results import SpeechClipSet
from ..protocols import (
    SessionSettings,
)
from .indeterminate_speaker_assignment import IndeterminantSpeakerAssignmentCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class MarkBackchannelsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Mark Backchannels"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.indeterminate_speakers)
        self.outputs.append(session_dir / settings.paths.backchannel_marked)
        self.dependencies.append(IndeterminantSpeakerAssignmentCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.indeterminate_speakers)
        output_clips: SpeechClipSet = mark_backchannels(
            settings, session_dir, input_clips, self, self.logger, self.tracer
        )
        self.save_speech_clip(output_clips, session_dir, settings.paths.backchannel_marked)
