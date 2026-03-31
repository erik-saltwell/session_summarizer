from __future__ import annotations

from pathlib import Path

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..vad import NemoVadDetector, SegmentSplitResult, SegmentSplitResultSet, compute_segments


def compute_vad_segments(
    settings: SessionSettings,
    session_dir: Path,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SegmentSplitResultSet:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing.

    Args:
        mode: "short" uses min/max_segment_length_short (for Canary transcription);
              "long" uses min/max_segment_length_long (for OOM-sensitive operations
              such as diarization).
    """
    final_path: Path = session_dir / settings.segments_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SegmentSplitResultSet.load(final_path)

    gpu_logger.report_gpu_usage("before VAD")

    detector: NemoVadDetector
    with logger.status("Loading VAD model."):
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

    vad_result = detector.detect(session_dir / settings.cleaned_audio_file, logger)
    gpu_logger.report_gpu_usage("after VAD")

    short_segments: SegmentSplitResult
    long_segments: SegmentSplitResult
    with logger.status("Computing segment cut points."):
        short_segments = compute_segments(
            vad_result,
            min_length=settings.min_segment_length_short,
            max_length=settings.max_segment_length_short,
        )
        long_segments = compute_segments(
            vad_result, min_length=settings.min_segment_length_long, max_length=settings.max_segment_length_long
        )

    logger.report_message("[blue]Comput segments complete.[/blue]")

    return SegmentSplitResultSet(short=short_segments, long=long_segments)
