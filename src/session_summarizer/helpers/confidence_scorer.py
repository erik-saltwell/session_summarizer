from __future__ import annotations

from pathlib import Path

from ..processing_results import AlignmentResult
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import ParakeetCTCConfidenceScorer
from ..vad import SegmentSplitResultSet

_PAUSE_THRESHOLD_S = 0.5  # gap between words that triggers a new segment
_MAX_SEGMENT_DURATION_S = 3.0  # hard cap on segment length
_SENTENCE_ENDERS = frozenset(".?!")


def score_confidence(
    settings: SessionSettings,
    session_dir: Path,
    aligned_transcription: AlignmentResult,
    segments: SegmentSplitResultSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> AlignmentResult:
    logger.report_message("[blue]Creating confidence scores.[/blue]")
    audio_path: Path = session_dir / settings.paths.cleaned_audio

    gpu_logger.report_gpu_usage("before processing")

    scorer: ParakeetCTCConfidenceScorer
    with logger.status("Creating scorer."):
        scorer = ParakeetCTCConfidenceScorer(device=settings.device)
        gpu_logger.report_gpu_usage("Created aligner")

    scored_alignment: AlignmentResult
    with logger.status("Scoring confidence."):
        scored_alignment = scorer.score(audio_path, aligned_transcription, segments, logger)

    gpu_logger.report_gpu_usage("after alignment")

    logger.report_message("[blue]Alignment complete.[/blue]")
    return scored_alignment
