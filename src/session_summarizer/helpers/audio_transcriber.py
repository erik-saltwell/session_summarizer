from __future__ import annotations

from pathlib import Path

from ..processing_results import TranscriberProtocol, TranscriptionResult
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import CanaryQwenTranscriber
from ..utils.tracer import Tracer
from ..vad import SegmentSplitResultSet


def transcribe_from_cleaned_audio(
    settings: SessionSettings,
    session_dir: Path,
    segments: SegmentSplitResultSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> TranscriptionResult:
    audio_path: Path = session_dir / settings.paths.cleaned_audio

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    gpu_logger.report_gpu_usage("before processing")

    transcriber: TranscriberProtocol
    with logger.status("Creating transcriber."):
        transcriber = CanaryQwenTranscriber(device=settings.device)
        gpu_logger.report_gpu_usage("Created transcriber")

    with logger.status("Transcribing..."):
        result: TranscriptionResult = transcriber.transcribe(audio_path, segments, logger, tracer)

    gpu_logger.report_gpu_usage("After transcription")
    tracer.add_context("Transcribed text length", len(result.plain_text()))

    return result
