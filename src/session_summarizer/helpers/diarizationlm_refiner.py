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
    diarized_clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    model = DiarizationLMModel(device=settings.device)
    model.load()
    try:
        processor = DiarizationLMProcessor(model)
        result = processor.process(diarized_clips, settings.epsilon)
    finally:
        model.unload()

    return result
