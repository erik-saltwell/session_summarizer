from __future__ import annotations

from pathlib import Path

from session_summarizer.processing_results.speech_clip_set import SpeechClip, SpeechClipFlags
from session_summarizer.settings.diarization_stitching_settings import DiarizationStitchingSettings

from ..diarization import MergeSelector, MergeType
from ..diarization.clip_merger import clips_are_close_enough, merge_clips, second_clip_is_superset
from ..processing_results.speech_clip_set import SpeechClipSet
from ..protocols import (
    GpuLogger,
    LoggingProtocol,
    SessionSettings,
)


class MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: DiarizationStitchingSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if prior_clip.has_flag(SpeechClipFlags.END_OF_TURN):
            return MergeType.NO_MERGE
        if not second_clip_is_superset(prior_clip, current_clip, settings, True, logger):
            return MergeType.NO_MERGE
        if not clips_are_close_enough(
            prior_clip,
            current_clip,
            settings.unfinished_clip_merge_max_length,
            settings.epsilon,
            logger,
        ):
            return MergeType.NO_MERGE

        return MergeType.MERGE_WITH_PRIOR


def apply_first_stitching(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    use_cache_if_present: bool,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    final_path: Path = session_dir / settings.first_stitched_path
    if final_path.exists() and use_cache_if_present:
        logger.report_message(f"[yellow]{final_path} already exists, returning cached instance.[/yellow]")
        return SpeechClipSet.load_from_json(final_path)

    merge_selector: MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous = (
        MergeUnfinishedSegmentsWithSameSpeakerOrAnonymous()
    )
    merged_clips = merge_clips(clips, merge_selector, settings.diarization_stitching, logger)

    return merged_clips
