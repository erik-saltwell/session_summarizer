from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..diarizationlm import DiarizationLMModel, DiarizationLMProcessor
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


@dataclass(frozen=True)
class DiarizationLMResult:
    clips: SpeechClipSet
    prompt_segment_count: int | None
    prompt_word_count: int | None


def apply_diarizationlm(
    settings: SessionSettings,
    session_dir: Path,
    diarized_clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> DiarizationLMResult:
    model = DiarizationLMModel(device=settings.device)
    model.load()
    try:
        processor = DiarizationLMProcessor(model)
        clips = processor.process(diarized_clips, settings.epsilon)
        result = DiarizationLMResult(
            clips=clips,
            prompt_segment_count=processor.last_prompt_segment_count,
            prompt_word_count=processor.last_prompt_word_count,
        )
    finally:
        model.unload()

    return result
