from __future__ import annotations

from pathlib import Path

from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
    TranscriberProtocol,
    TranscriptionResult,
)
from ..transcription import CanaryQwenTranscriber
from ..vad import SegmentSplitResultSet


def transcribe_from_cleaned_audio(
    settings: SessionSettings,
    session_dir: Path,
    segments: SegmentSplitResultSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> TranscriptionResult:
    original_path: Path = session_dir / settings.cleaned_audio_file
    final_path: Path = session_dir / settings.transcript_file

    logger.report_message(f"[blue]Transcribing from clean audio at {original_path}[/blue]")
    if final_path.exists():
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return TranscriptionResult.load(final_path)

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    gpu_logger.report_gpu_usage("before processing")

    transcriber: TranscriberProtocol
    with logger.status("Creating transcriber."):
        transcriber = CanaryQwenTranscriber(device=settings.device)
        gpu_logger.report_gpu_usage("Created transcriber")

    result: TranscriptionResult = transcriber.transcribe(original_path, segments, logger)
    gpu_logger.report_gpu_usage("After transcription")

    logger.report_message("[blue]Transcription complete.[/blue]")
    return result
