from __future__ import annotations

from pathlib import Path

from session_summarizer.processing_results.speech_clip import SpeechClip

from ..diarization import MergeSelector, MergeType
from ..diarization.clip_merger import (
    clips_are_close_enough,
    merge_clips,
)
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


class IdentityMergeSelector(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: SessionSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if prior_clip.identity is None or current_clip.identity is None:
            return MergeType.NO_MERGE

        if prior_clip.identity != current_clip.identity:
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.stitching.identity_merge_max_gap_seconds,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_WITH_PRIOR


def apply_identity_stitching(
    settings: SessionSettings,
    session_dir: Path,
    identified_clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    merge_selector: IdentityMergeSelector = IdentityMergeSelector()
    merged_clips = merge_clips(identified_clips, merge_selector, settings, logger)

    return merged_clips
