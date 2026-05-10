from __future__ import annotations

from pathlib import Path
from typing import cast

from ..evaluation.text_cleaner import clean_text_for_evaluation
from ..processing_results import SpeechClipSet
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings
from ..utils import Tracer


def punctuate_text(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    from punctuators.models import PunctCapSegModelONNX

    indexed_segments: list[tuple[int, str]] = [
        (i, clean_text_for_evaluation(clip.text, do_mathspell=False)) for i, clip in enumerate(clips) if clip.words
    ]
    non_empty: list[tuple[int, str]] = [(i, s) for i, s in indexed_segments if s]
    if non_empty:
        clip_indices, segments = zip(*non_empty, strict=True)
        model = PunctCapSegModelONNX.from_pretrained("pcs_en")
        results_tmp = model.infer(texts=list(segments), apply_sbd=False)
        results = cast(list[str], results_tmp)
        for clip_idx, result in zip(clip_indices, results, strict=True):
            clips[clip_idx].text = result
    return clips
