from __future__ import annotations

from pathlib import Path

from ..processing_results import SpeechClipSet
from ..protocols import GpuLogger, LoggingProtocol
from ..settings import SessionSettings


def assign_utterance_ids(
    settings: SessionSettings,
    session_dir: Path,
    clips: SpeechClipSet,
    gpu_logger: GpuLogger,
    logger: LoggingProtocol,
) -> SpeechClipSet:
    campaign_id = settings.campaign_info.campaign_id
    session_id = settings.session_info.session_id

    clips.sort_clips()
    for n, clip in enumerate(clips, start=1):
        clip.utterance_id = f"{campaign_id}_{session_id}_{n}"
    logger.report_message(f"Assigned {len(clips)} utterance ids with prefix {campaign_id}_{session_id}_")
    return clips
