from __future__ import annotations

from pathlib import Path

from ..processing_results import AlignmentResult, TranscriptionResult
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import ParakeetCTCWordAligner
from ..utils import Tracer
from ..vad import SegmentSplitResultSet

_PAUSE_THRESHOLD_S = 0.5  # gap between words that triggers a new segment
_MAX_SEGMENT_DURATION_S = 3.0  # hard cap on segment length
_SENTENCE_ENDERS = frozenset(".?!")


def align_transcript(
    settings: SessionSettings,
    session_dir: Path,
    transcription: TranscriptionResult,
    segments: SegmentSplitResultSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> AlignmentResult:
    gpu_logger.report_gpu_usage("before processing")

    aligner: ParakeetCTCWordAligner
    with logger.status("Creating aligner."):
        aligner = ParakeetCTCWordAligner(device=settings.device)
        gpu_logger.report_gpu_usage("Created aligner")

    alignment: AlignmentResult = aligner.align(
        session_dir / settings.paths.cleaned_audio, transcription, segments, logger, tracer
    )
    gpu_logger.report_gpu_usage("after alignment")

    return alignment
