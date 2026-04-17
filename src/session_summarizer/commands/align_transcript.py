from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths
from session_summarizer.commands.clean_audio import CleanAudioCommand

from ..helpers.audio_segmenter import SegmentSplitResultSet
from ..helpers.transcript_aligner import align_transcript
from ..processing_results import AlignmentResult, TranscriptionResult
from ..settings.session_settings import SessionSettings
from .compute_segments import ComputeSegmentsCommand
from .session_processing_command import SessionProcessingCommand
from .transcribe_audio import TranscribeAudioCommand


@dataclass
class AlignTranscriptCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Align Transcript"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.paths.transcript)
        self.inputs.append(session_dir / settings.paths.vad_segments)
        self.inputs.append(session_dir / settings.paths.cleaned_audio)
        self.outputs.append(session_dir / settings.paths.aligned_transcript)
        self.dependencies.append(TranscribeAudioCommand(self.session_id, self.tracer))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id, self.tracer))
        self.dependencies.append(CleanAudioCommand(self.session_id, self.tracer))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        segments: SegmentSplitResultSet = SegmentSplitResultSet.load(session_dir / settings.paths.vad_segments)
        transcript: TranscriptionResult = TranscriptionResult.load_from_json(session_dir / settings.paths.transcript)

        alignment: AlignmentResult = align_transcript(settings, session_dir, transcript, segments, self, self.logger)
        self.save_alignment_result(alignment, session_dir, settings.paths.aligned_transcript)
