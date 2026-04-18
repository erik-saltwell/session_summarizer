from __future__ import annotations

from math import isnan
from pathlib import Path

from ..processing_results import SpeechClip, SpeechClipSet
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings
from ..utils import Tracer
from .unassigned_speaker import UNASSIGNED_SPEAKER_NAME


def _should_unassign_speaker(clip: SpeechClip, settings: SessionSettings) -> bool:
    if clip.similarity_residual is None:
        return False

    if (
        isnan(clip.similarity_residual)
        or clip.similarity_residual < settings.speaker_identification.assignment_threshold
    ):
        return True

    return False


def assign_indeterminate_speakers(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
    tracer: Tracer,
) -> SpeechClipSet:
    unassigned_count: int = 0
    for clip in clips:
        if _should_unassign_speaker(clip, settings):
            unassigned_count += 1
            clip.identity = UNASSIGNED_SPEAKER_NAME
    tracer.add_context("unassigned_count", unassigned_count)
    return clips
