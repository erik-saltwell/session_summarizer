from __future__ import annotations

from pathlib import Path

from ..diarizationlm import DiarizationLMModel, DiarizationLMProcessor
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


def apply_diarizationlm(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.diarizationlm_processed_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    model = DiarizationLMModel(device=settings.device)
    model.load()
    try:
        processor = DiarizationLMProcessor(model)
        result = processor.process(clips)
    finally:
        model.unload()

    return result
