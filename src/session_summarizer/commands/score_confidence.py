from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.audio_segmenter import SegmentSplitResultSet
from ..helpers.confidence_scorer import score_confidence
from ..settings.session_settings import SessionSettings
from ..transcription.parakeet_ctc_confidence_scorer import AlignmentResult
from .align_transcript import AlignTranscriptCommand
from .clean_audio import CleanAudioCommand
from .compute_segments import ComputeSegmentsCommand
from .session_processing_command import SessionProcessingCommand


@dataclass
class ScoreConfidenceCommand(SessionProcessingCommand):
    def name(self) -> str:
        return "Score Confidence"

    def add_dependencies(self, settings: SessionSettings, session_dir: Path) -> None:
        self.inputs.append(session_dir / settings.aligned_transcript_path)
        self.inputs.append(session_dir / settings.segments_path)
        self.outputs.append(session_dir / settings.confidence_transcript_path)
        self.dependencies.append(AlignTranscriptCommand(self.session_id))
        self.dependencies.append(ComputeSegmentsCommand(self.session_id))
        self.dependencies.append(CleanAudioCommand(self.session_id))

    def process_session(self, settings: SessionSettings, session_dir: common_paths.Path) -> None:
        segments: SegmentSplitResultSet = SegmentSplitResultSet.load(session_dir / settings.segments_path)
        alignment: AlignmentResult = AlignmentResult.load_from_json(session_dir / settings.aligned_transcript_path)

        scored_alignment = score_confidence(settings, session_dir, alignment, segments, self, self.logger)
        self.save_alignment_result(scored_alignment, session_dir, settings.confidence_transcript_path)
