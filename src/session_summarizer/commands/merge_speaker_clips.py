from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..audio.speaker_tools import merge_speaker_clips_to_min_duration
from ..protocols import LoggingProtocol, NullLogger
from ..settings.session_settings import SessionSettings

_SETTINGS_FILE = "settings.yaml"


@dataclass
class MergeSpeakerClipsCommand:
    speaker_label: str
    output_folder: Path
    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Merge Speaker Clips"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        settings = SessionSettings.load(common_paths.data_dir() / _SETTINGS_FILE)
        min_duration = settings.minimum_speaker_clip_duration

        input_dir = common_paths.voice_samples_for_speaker(self.speaker_label)
        if not input_dir.exists():
            raise FileNotFoundError(f"Speaker folder not found: {input_dir}")

        logger.report_message(
            f"[blue]Merging clips for '{self.speaker_label}' (min duration: {min_duration:.2f}s)[/blue]"
        )
        merge_speaker_clips_to_min_duration(
            input_dir, self.output_folder, min_duration, settings.speaker_clip_gap_length, logger
        )
