from __future__ import annotations

from pathlib import Path

from session_summarizer.diarization.diarizen_diarizer import DiarizationResult, DiarizenDiarizer
from session_summarizer.transcription.parakeet_ctc_confidence_scorer import AlignmentResult
from session_summarizer.vad.segment_splitter import SegmentSplitResultSet

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


def diarize_audio(
    settings: SessionSettings,
    session_dir: Path,
    alignment: AlignmentResult,
    segments: SegmentSplitResultSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> DiarizationResult:
    logger.report_message("[blue]Diarizing audio.[/blue]")
    final_path: Path = session_dir / settings.base_diarized_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return DiarizationResult.load(final_path)

    gpu_logger.report_gpu_usage("before processing")

    diarizer: DiarizenDiarizer = DiarizenDiarizer()
    result: DiarizationResult = diarizer.diarize(session_dir / settings.cleaned_audio_file, logger)
    logger.report_message("[blue]Diarization complete.[/blue]")

    return result
