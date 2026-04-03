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
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    logger.report_message("[blue]Diarizing audio.[/blue]")
    final_path: Path = session_dir / settings.base_diarized_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    gpu_logger.report_gpu_usage("before processing")

    diarizer: DiarizenDiarizer = DiarizenDiarizer()
    diarization: MergedDiarizationResult = diarizer.diarize(session_dir / settings.cleaned_audio_file, logger)
    logger.report_message(f"[blue]Converting to SpeechClipSet {final_path}...[/blue]")
    result: SpeechClipSet = create_speech_clips(diarization, alignment_result, settings, logger)

    logger.report_message("[blue]Diarization complete.[/blue]")

    return result
