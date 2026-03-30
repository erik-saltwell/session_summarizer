from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings
from session_summarizer.transcription.parakeet_ctc_confidence_scorer import AlignmentResult

from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.transcript_aligner import align_transcript
from ..protocols.transcriber_protocol import TranscriptionResult
from .session_processing_command import SessionProcessingCommand


@dataclass
class ScoreConfidenceCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Align Transcript"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clean_audio(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(settings, session_dir, True, self, self.logger)
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, True, self, self.logger)
        alignment = score_confidence(settings, session_dir, alignment, False, self, self.logger)
        alignment.save(session_dir / settings.confidence_transcript_path)
