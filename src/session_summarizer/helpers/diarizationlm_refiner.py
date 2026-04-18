from __future__ import annotations

from pathlib import Path

from ..diarizationlm import DiarizationLMModel, DiarizationLMProcessor
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)
from ..utils import Tracer, silence_os_noise


def apply_diarizationlm(
    settings: SessionSettings,
    session_dir: Path,
    diarized_clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    model = DiarizationLMModel(device=settings.device)
    with logger.status("Loading model..."):
        with silence_os_noise():
            model.load()
    try:
        with logger.status("Running diarizationlm..."):
            processor = DiarizationLMProcessor(model)
            result = processor.process(diarized_clips, settings.epsilon, tracer)

    finally:
        model.unload()

    return result
