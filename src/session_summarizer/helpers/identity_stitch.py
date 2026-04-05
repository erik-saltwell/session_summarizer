from __future__ import annotations

from pathlib import Path

from session_summarizer.processing_results.speech_clip_set import SpeechClip
from session_summarizer.settings.diarization_stitching_settings import DiarizationStitchingSettings

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


class IdentityBackchannelMerger(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: DiarizationStitchingSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if (
            prior_clip.identity is None
            or current_clip.identity is None
            or next_clip is None
            or next_clip.identity is None
        ):
            return MergeType.NO_MERGE

        if prior_clip.identity == current_clip.identity:
            return MergeType.NO_MERGE

        if not prior_clip.identity == next_clip.identity:
            return MergeType.NO_MERGE

        if current_clip.duration > settings.max_identity_backchannel_duration:
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.max_identity_backchannel_prior_gap,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            current_clip,
            next_clip,
            settings.max_identity_backchannel_next_gap,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_ALL_THREE


class IdentityMergeSelector(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: DiarizationStitchingSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if prior_clip.identity is None or current_clip.identity is None:
            return MergeType.NO_MERGE

        if prior_clip.identity != current_clip.identity:
            return MergeType.NO_MERGE

        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.identity_stitching_max_gap,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_WITH_PRIOR


def apply_identity_stitching(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.identity_stitched_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    merge_selector: IdentityMergeSelector = IdentityMergeSelector()
    merged_clips = merge_clips(clips, merge_selector, settings.diarization_stitching, logger)

    backchannel_merge_selector: IdentityBackchannelMerger = IdentityBackchannelMerger()
    merged_clips = merge_clips(merged_clips, backchannel_merge_selector, settings.diarization_stitching, logger)

    return merged_clips
