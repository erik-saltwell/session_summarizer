from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.indeterminate_speakers import assign_indeterminate_speakers
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .session_processing_command import SessionProcessingCommand
from .stitch_identities import StitichIdentitiesCommand


@dataclass
class IndeterminantSpeakerAssignmentCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Indeterminant Speakers"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.identity_stitched_path)
        self.outputs.append(session_dir / settings.indeterminate_speakers_path)
        self.dependencies.append(StitichIdentitiesCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        input_clips: SpeechClipSet = SpeechClipSet.load_from_json(session_dir / settings.identity_stitched_path)

        output_clips: SpeechClipSet = assign_indeterminate_speakers(
            settings, session_dir, input_clips, self, self.logger
        )

        self.save_speech_clip(output_clips, session_dir, settings.indeterminate_speakers_path)
