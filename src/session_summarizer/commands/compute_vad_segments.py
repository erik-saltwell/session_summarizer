from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings

from ..helpers.audio_cleaner import clean_audio
from ..helpers.vad_segmenter import compute_vad_segments
from .session_processing_command import SessionProcessingCommand


@dataclass
class ComputeVadSegmentsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Compute VAD Segments"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clean_audio(settings, session_dir, True, self, self.logger)
        compute_vad_segments(settings, session_dir, False, self, self.logger)
