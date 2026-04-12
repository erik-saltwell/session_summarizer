from __future__ import annotations

from pathlib import Path

from ..diarization import MergeSelector, MergeType
from ..diarization.clip_merger import (
    clips_are_close_enough,
    clips_are_same_speaker,
    clips_have_subset_superset_relationship,
    merge_clips,
)
from ..processing_results import SpeechClip, SpeechClipFlags, SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


class BackchannelMerger(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: SessionSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if clips_are_same_speaker(prior_clip, current_clip, settings, True, logger):
            return MergeType.NO_MERGE
        if next_clip is None:
            return MergeType.NO_MERGE

        if not clips_are_same_speaker(prior_clip, next_clip, settings, True, logger):
            return MergeType.NO_MERGE

        if current_clip.duration > settings.diarization_stitching.max_backchannel_duration:
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.diarization_stitching.max_backchannel_prior_gap,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            current_clip,
            next_clip,
            settings.diarization_stitching.max_backchannel_next_gap,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_ALL_THREE


class MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: SessionSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if prior_clip.has_flag(SpeechClipFlags.END_OF_TURN):
            return MergeType.NO_MERGE
        if not clips_have_subset_superset_relationship(prior_clip, current_clip, settings, True, logger):
            return MergeType.NO_MERGE
        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.diarization_stitching.unfinished_clip_merge_max_length,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_WITH_PRIOR


def apply_first_stitching(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    backchannel_selector: BackchannelMerger = BackchannelMerger()
    merged_clips = merge_clips(clips, backchannel_selector, settings, logger)

    merge_selector: MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous = (
        MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous()
    )
    merged_clips = merge_clips(merged_clips, merge_selector, settings, logger)

    return merged_clips
