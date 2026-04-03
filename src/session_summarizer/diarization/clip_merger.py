from __future__ import annotations

from enum import IntEnum, auto
from typing import Protocol

from session_summarizer.protocols.logging_protocol import LoggingProtocol
from session_summarizer.settings.diarization_stitching_settings import DiarizationStitchingSettings

from ..processing_results import SpeechClip, SpeechClipSet


class MergeType(IntEnum):
    NO_MERGE = 0
    MERGE_WITH_PRIOR = auto()
    MERGE_WITH_NEXT = auto()
    MERGE_ALL_THREE = auto()


class MergeSelector(Protocol):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: DiarizationStitchingSettings,
        logger: LoggingProtocol,
    ) -> MergeType: ...


def merge_segments(
    initial_clips: SpeechClipSet,
    selector: MergeSelector,
    settings: DiarizationStitchingSettings,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    if len(initial_clips) <= 1:
        return SpeechClipSet()

    merged_clips: SpeechClipSet = SpeechClipSet()
    merged_clips.add_clip(initial_clips[0])
    i: int = 1
    while i < len(initial_clips):
        prior_clip: SpeechClip = merged_clips[-1]
        current_clip: SpeechClip = initial_clips[i]
        next_clip: SpeechClip | None = initial_clips[i + 1] if i + 1 < len(initial_clips) else None
        merge_type: MergeType = selector.ShouldMerge(prior_clip, current_clip, next_clip, settings, logger)

        if merge_type == MergeType.NO_MERGE:
            merged_clips.add_clip(current_clip)
            i += 1
        elif merge_type == MergeType.MERGE_WITH_PRIOR:
            prior_clip.merge(current_clip)
            i += 1
        elif merge_type == MergeType.MERGE_WITH_NEXT:
            if next_clip is None:
                raise RuntimeError("trying to merge with next but at last clip")
            current_clip.merge(next_clip)
            merged_clips.add_clip(current_clip)
            i += 2
        elif merge_type == MergeType.MERGE_ALL_THREE:
            if next_clip is None:
                raise RuntimeError("trying to merge with previous and next but at last clip")
            prior_clip.merge(current_clip)
            prior_clip.merge(next_clip)
            i += 2
        else:
            raise RuntimeError("Unexpected Merge Type")
    merged_clips.sort_clips()
    return merged_clips
