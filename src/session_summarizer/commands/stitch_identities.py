from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.identity_stitch import apply_identity_stitching
from ..processing_results import SpeechClipSet
from ..settings import SessionSettings
from .identify_speakers import IdentifySpeakersCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class StitichIdentitiesCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Stitch Identities"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.identified_speakers)
        self.outputs.append(session_dir / settings.paths.identity_stitched)
        self.dependencies.append(IdentifySpeakersCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        identified_speaker_clips: SpeechClipSet = SpeechClipSet.load_from_json(
            session_dir / settings.paths.identified_speakers
        )

        id_stitched_clips: SpeechClipSet = apply_identity_stitching(
            settings, session_dir, identified_speaker_clips, self, self.logger, self.tracer
        )
        self.tracer.add_context("pre_stitch_segment_count", len(identified_speaker_clips))
        self.tracer.add_context("post_stitch_segment_count", len(id_stitched_clips))

        self.save_speech_clip(id_stitched_clips, session_dir, settings.paths.identity_stitched)
