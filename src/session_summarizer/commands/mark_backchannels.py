from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..helpers.backchannel_marker import is_backchannel
from ..processing_results import SpeechClipFlags, SpeechClipSet
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
        clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.paths.indeterminate_speakers)
        backchannel_count = 0
        for clip in clips:
            is_clip_backchannel = is_backchannel(clip)
            clip.set_flag(SpeechClipFlags.IS_BACKCHANNEL, is_clip_backchannel)
            if is_clip_backchannel:
                backchannel_count += 1
        self.logger.report_message(f"Marked {backchannel_count} out of {len(clips)} clips as backchannels")
        self.save_speech_clip(clips, session_dir, settings.paths.backchannel_marked)
