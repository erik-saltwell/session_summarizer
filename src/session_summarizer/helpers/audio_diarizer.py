from __future__ import annotations

from pathlib import Path

from ..diarization import DiarizenDiarizer, MergedDiarizationResult, create_speech_clips
from ..processing_results import AlignmentResult, SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..utils import Tracer


def diarize_audio(
    settings: SessionSettings,
    session_dir: Path,
    alignment_result: AlignmentResult,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    audio_path = session_dir / settings.paths.cleaned_audio

    gpu_logger.report_gpu_usage("before processing")

    with logger.status("Diarizing..."):
        diarizer: DiarizenDiarizer = DiarizenDiarizer()
        diarization: MergedDiarizationResult = diarizer.diarize(audio_path, len(settings.attendees), logger, tracer)
    with logger.status("Creating speech clips from diarization..."):
        result: SpeechClipSet = create_speech_clips(diarization, alignment_result, settings, logger, tracer)

    return result
