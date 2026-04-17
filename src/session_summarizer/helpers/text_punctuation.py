from __future__ import annotations

from pathlib import Path
from typing import cast

from punctuators.models import PunctCapSegModelONNX

from ..evaluation.text_cleaner import clean_text_for_evaluation
from ..processing_results import SpeechClipSet
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings


def punctuate_text(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    segments: list[str] = [clean_text_for_evaluation(clip.text) for clip in clips]
    model = PunctCapSegModelONNX.from_pretrained("pcs_en")
    results = cast(list[str], model.infer(texts=segments, apply_sbd=False))
    for idx, result in enumerate(results):
        clips[idx].text = result
    return clips
