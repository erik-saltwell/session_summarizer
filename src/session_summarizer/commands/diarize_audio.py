from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.helpers.audio_diarizer import diarize_audio
from session_summarizer.processing_results.speech_clip_set import SpeechClipSet
from session_summarizer.settings.session_settings import SessionSettings
from session_summarizer.transcription.parakeet_ctc_confidence_scorer import AlignmentResult

from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.transcript_aligner import align_transcript
from ..protocols.transcriber_protocol import TranscriptionResult
from .session_processing_command import SessionProcessingCommand


@dataclass
class DiarizeAudioCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Diarize Audio"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = False
        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, True, self, self.logger
        )
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, segments, True, self, self.logger)
        alignment = score_confidence(settings, session_dir, alignment, segments, True, self, self.logger)
        clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, False, self, self.logger)
        clips.save_to_json(session_dir / settings.base_diarized_path)
