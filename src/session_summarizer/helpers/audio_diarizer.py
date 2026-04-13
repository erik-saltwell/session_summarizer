from __future__ import annotations

from pathlib import Path

from ..diarization import DiarizenDiarizer, MergedDiarizationResult, create_speech_clips
from ..processing_results import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..transcription import AlignmentResult


def diarize_audio(
    settings: SessionSettings,
    session_dir: Path,
    alignment_result: AlignmentResult,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    logger.report_message("[blue]Diarizing audio.[/blue]")
    final_path: Path = session_dir / settings.paths.base_diarization
    audio_path = session_dir / settings.paths.cleaned_audio

    gpu_logger.report_gpu_usage("before processing")

    diarizer: DiarizenDiarizer = DiarizenDiarizer()
    diarization: MergedDiarizationResult = diarizer.diarize(audio_path, logger)
    logger.report_message(f"[blue]Converting to SpeechClipSet {final_path}...[/blue]")
    result: SpeechClipSet = create_speech_clips(diarization, alignment_result, settings, logger)

    logger.report_message("[blue]Diarization complete.[/blue]")

    return result
