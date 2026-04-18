from __future__ import annotations

from pathlib import Path

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..utils import Tracer
from ..vad import NemoVadDetector, SegmentSplitResult, SegmentSplitResultSet, compute_segments


def compute_vad_segments(
    settings: SessionSettings,
    session_dir: Path,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SegmentSplitResultSet:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing.

    Args:
        mode: "short" uses min/max_segment_length_short (for Canary transcription);
              "long" uses min/max_segment_length_long (for OOM-sensitive operations
              such as diarization).
    """

    gpu_logger.report_gpu_usage("before VAD")

    detector: NemoVadDetector
    with logger.status("Loading VAD model..."):
        detector = NemoVadDetector(
            model_name=settings.vad.model_name,
            device=settings.device,
            onset=settings.vad.onset,
            offset=settings.vad.offset,
            min_duration_on=settings.vad.min_duration_on,
            min_duration_off=settings.vad.min_duration_off,
            pad_onset=settings.vad.pad_onset,
            pad_offset=settings.vad.pad_offset,
        )

    with logger.status("Segmenting with model"):
        vad_result = detector.detect(session_dir / settings.paths.cleaned_audio, logger, tracer)

    gpu_logger.report_gpu_usage("after VAD")

    short_segments: SegmentSplitResult
    long_segments: SegmentSplitResult
    with logger.status("Computing segment cut points."):
        short_segments = compute_segments(
            vad_result,
            min_length=settings.segmentation.short_min_seconds,
            max_length=settings.segmentation.short_max_seconds,
        )
        long_segments = compute_segments(
            vad_result,
            min_length=settings.segmentation.long_min_seconds,
            max_length=settings.segmentation.long_max_seconds,
        )

    tracer.add_context("short_segment_count", len(short_segments.segments))
    tracer.add_context("long_segment_count", len(long_segments.segments))
    tracer.add_context("longest_short_segment", max(segment.duration for segment in short_segments.segments))
    tracer.add_context("longest_long_segment", max(segment.duration for segment in long_segments.segments))

    return SegmentSplitResultSet(short=short_segments, long=long_segments)
