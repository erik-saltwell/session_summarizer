from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings

from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..protocols.transcriber_protocol import TranscriptionResult
from .session_processing_command import SessionProcessingCommand


@dataclass
class TranscribeAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Transcribe"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = False
        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, False, self, self.logger
        )
        result.save_to_json(session_dir / settings.transcript_file)
