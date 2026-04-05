from __future__ import annotations

from dataclasses import dataclass

import session_summarizer.utils.common_paths as common_paths

from ..helpers.add_embeddings import add_embeddings
from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_diarizer import diarize_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.first_stitcher import apply_first_stitching
from ..helpers.speaker_identifier import identify_speakers
from ..helpers.transcript_aligner import align_transcript
from ..helpers.update_turn_end import update_turn_end
from ..processing_results import AlignmentResult, SpeechClipSet, TranscriptionResult
from ..settings import SessionSettings
from .session_processing_command import SessionProcessingCommand


@dataclass
class IdentifySpeakersCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Identify Speakers"

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        clean_audio(settings, session_dir, True, self, self.logger)
        segments: SegmentSplitResultSet = compute_vad_segments(settings, session_dir, True, self, self.logger)
        result: TranscriptionResult = transcribe_from_cleaned_audio(
            settings, session_dir, segments, True, self, self.logger
        )
        alignment: AlignmentResult = align_transcript(settings, session_dir, result, segments, True, self, self.logger)
        alignment = score_confidence(settings, session_dir, alignment, segments, True, self, self.logger)
        diarized_clips: SpeechClipSet = diarize_audio(settings, session_dir, alignment, True, self, self.logger)

        turn_clips: SpeechClipSet = update_turn_end(settings, session_dir, diarized_clips, True, self, self.logger)

        stitched_clips: SpeechClipSet = apply_first_stitching(
            settings, session_dir, turn_clips, True, self, self.logger
        )
        embedded_clips: SpeechClipSet = add_embeddings(settings, session_dir, stitched_clips, True, self, self.logger)
        identified_speaker_clips: SpeechClipSet = identify_speakers(
            settings, session_dir, embedded_clips, False, self, self.logger
        )
        identified_speaker_clips.save_to_json(session_dir / settings.identified_speaker_path)
