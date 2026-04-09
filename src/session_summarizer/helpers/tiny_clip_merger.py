from __future__ import annotations

from pathlib import Path

from ..diarization.clip_merger import MergeSelector, MergeType, merge_clips
from ..processing_results import SpeechClip, SpeechClipSet
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings


class TinyClipMergeSelector(MergeSelector):
    def ShouldMerge(
        self,
        prior_clip: SpeechClip,
        current_clip: SpeechClip,
        next_clip: SpeechClip | None,
        settings: SessionSettings,
        logger: LoggingProtocol,
    ) -> MergeType:
        if current_clip.duration > settings.diarization_stitching.tiny_clip_threshold:
            return MergeType.NO_MERGE

        if next_clip is not None and current_clip.gap_distance(next_clip, settings.epsilon) < current_clip.gap_distance(
            prior_clip, settings.epsilon
        ):
            return MergeType.MERGE_WITH_NEXT

        return MergeType.MERGE_WITH_PRIOR


def apply_tiny_stitching(
    settings: SessionSettings,
    session_dir: Path,
    identified_clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    merge_selector: TinyClipMergeSelector = TinyClipMergeSelector()
    merged_clips = merge_clips(identified_clips, merge_selector, settings, logger)

    return merged_clips
