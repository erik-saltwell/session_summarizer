from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.commands.clean_audio import CleanAudioCommand

from ..helpers.audio_segmenter import SegmentSplitResultSet
from ..helpers.transcript_aligner import align_transcript
from ..processing_results.transcriber_protocol import TranscriptionResult
from ..settings.session_settings import SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult
from .compute_segments import ComputeSegmentsCommand
from .session_processing_command import SessionProcessingCommand
from .transcribe_audio import TranscribeAudioCommand


@dataclass
class AlignTranscriptCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Align Transcript"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.transcript_file)
        self.inputs.append(session_dir / settings.segments_path)
        self.inputs.append(session_dir / settings.cleaned_audio_file)
        self.outputs.append(session_dir / settings.aligned_transcript_path)
        self.dependencies.append(TranscribeAudioCommand(self.session_id))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id))
        self.dependencies.append(CleanAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        segments: SegmentSplitResultSet = SegmentSplitResultSet.load(session_dir / settings.segments_path)
        transcript: TranscriptionResult = TranscriptionResult.load_from_json(session_dir / settings.transcript_file)

        alignment: AlignmentResult = align_transcript(settings, session_dir, transcript, segments, self, self.logger)
        self.save_alignment_result(alignment, session_dir, settings.aligned_transcript_path)
