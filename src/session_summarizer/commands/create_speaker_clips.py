from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..audio.speaker_audio import save_segment_as_speaker_audio_clip
from ..helpers.add_embeddings import add_embeddings
from ..helpers.audio_cleaner import clean_audio
from ..helpers.audio_diarizer import diarize_audio
from ..helpers.audio_segmenter import SegmentSplitResultSet, compute_vad_segments
from ..helpers.audio_transcriber import transcribe_from_cleaned_audio
from ..helpers.confidence_scorer import score_confidence
from ..helpers.first_stitcher import apply_first_stitching
from ..helpers.identity_stitch import apply_identity_stitching
from ..helpers.speaker_identifier import identify_speakers
from ..helpers.transcript_aligner import align_transcript
from ..helpers.update_turn_end import update_turn_end
from ..processing_results import AlignmentResult, SpeechClipSet, TranscriptionResult
from ..protocols import (
    SessionSettings,
)
from .session_processing_command import SessionProcessingCommand


@dataclass
class CreateSpeakerClipsCommand(SessionProcessingCommand):
    use_multi_speaker_clips: bool = False

    def name(self) -> str:
        return "Create Speaker Clips"

    def process_session(self, settings: SessionSettings, session_dir: Path) -> None:
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
            settings, session_dir, embedded_clips, True, self, self.logger
        )

        id_stitched_clips: SpeechClipSet = apply_identity_stitching(
            settings, session_dir, identified_speaker_clips, True, self, self.logger
        )

        cleaned_audio_path = session_dir / settings.cleaned_audio_file

        saved_count = 0
        skipped_count = 0
        speaker_clip_counts: dict[str, int] = {}
        speaker_durations: dict[str, float] = {}

        for clip in id_stitched_clips:
            if clip.is_anonymous or clip.identity is None:
                skipped_count += 1
                continue
            if clip.is_multispeaker and not self.use_multi_speaker_clips:
                skipped_count += 1
                continue

            save_segment_as_speaker_audio_clip(
                cleaned_audio_path,
                clip,
                clip.identity,
                settings.speaker_clip_lead_in,
                settings.speaker_clip_lead_out,
            )
            saved_count += 1
            speaker_clip_counts[clip.identity] = speaker_clip_counts.get(clip.identity, 0) + 1
            speaker_durations[clip.identity] = speaker_durations.get(clip.identity, 0.0) + clip.duration

        self.report_message(f"Saved {saved_count} speaker clips, skipped {skipped_count}")

        if speaker_clip_counts:
            headers = ["Speaker", "Clips", "Duration (s)"]
            rows = [
                [speaker, str(speaker_clip_counts[speaker]), f"{speaker_durations[speaker]:.2f}"]
                for speaker in sorted(speaker_clip_counts)
            ]
            self.logger.report_multicolumn_table(headers, rows)
