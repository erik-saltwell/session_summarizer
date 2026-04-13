from __future__ import annotations

from pathlib import Path

from ..processing_results import TranscriberProtocol, TranscriptionResult
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import CanaryQwenTranscriber
from ..vad import SegmentSplitResultSet


def transcribe_from_cleaned_audio(
    settings: SessionSettings,
    session_dir: Path,
    segments: SegmentSplitResultSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> TranscriptionResult:
    audio_path: Path = session_dir / settings.paths.cleaned_audio

    logger.report_message(f"[blue]Transcribing from clean audio at {audio_path}[/blue]")

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    gpu_logger.report_gpu_usage("before processing")

    transcriber: TranscriberProtocol
    with logger.status("Creating transcriber."):
        transcriber = CanaryQwenTranscriber(device=settings.device)
        gpu_logger.report_gpu_usage("Created transcriber")

    result: TranscriptionResult = transcriber.transcribe(audio_path, segments, logger)
    gpu_logger.report_gpu_usage("After transcription")

    logger.report_message("[blue]Transcription complete.[/blue]")
    return result
