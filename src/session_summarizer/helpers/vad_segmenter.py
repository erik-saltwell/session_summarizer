from __future__ import annotations

from pathlib import Path

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..vad import NemoVadDetector, SegmentSplitResult, compute_segments


def compute_vad_segments(
    settings: SessionSettings,
    session_dir: Path,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SegmentSplitResult:
    """Run VAD on cleaned audio and compute optimal cut points for chunked processing."""
    final_path: Path = session_dir / settings.vad_segments_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SegmentSplitResult.load(final_path)

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

    with logger.status("Computing segment cut points."):
        result = compute_segments(
            vad_result,
            min_length=settings.min_segment_length,
            max_length=settings.max_segment_length,
        )

    logger.report_message(
        f"[blue]Computed {len(result.segments)} segments with {len(result.cut_points)} cut points[/blue]"
    )

    result.save(final_path)
    logger.report_message(f"[green]VAD segments saved to {final_path}[/green]")

    return result
