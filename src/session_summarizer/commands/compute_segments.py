from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_segmenter import compute_vad_segments
from ..settings.session_settings import SessionSettings
from ..vad import SegmentSplitResultSet
from .clean_audio import CleanAudioCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class ComputeSegmentsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Compute Segments"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.outputs.append(session_dir / settings.paths.vad_segments)
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        results: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, self, self.logger)
        results.save(session_dir / settings.paths.vad_segments)
