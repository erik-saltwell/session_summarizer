from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.protocols.session_settings import SessionSettings
from session_summarizer.transcription.parakeet_ctc_confidence_scorer import AlignmentResult

from ..evaluation import clean_text_for_evaluation
from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.transcript_aligner import align_transcript
from ..protocols.transcriber_protocol import TranscriptionResult
from .session_processing_command import SessionProcessingCommand


@dataclass
class DumpAndCompareTextsCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Dump and Compare Texts"

    def create_output_path(self, original_path: Path) -> Path:
        return original_path.parent / (original_path.stem + "_text.txt")

    def clean_and_dump_text(self, text: str, original_path: Path) -> str:
        cleaned_text = clean_text_for_evaluation(text)
        with open(self.create_output_path(original_path), "w") as f:
            f.write(cleaned_text)
        return cleaned_text

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        self.gpu_logging_enabled = True
        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, True, self, self.logger
        )
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, segments, True, self, self.logger)

        alignment = score_confidence(settings, session_dir, alignment, segments, False, self, self.logger)
        alignment.save_to_json(session_dir / settings.confidence_transcript_path)
