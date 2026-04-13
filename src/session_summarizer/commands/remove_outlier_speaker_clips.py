from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..helpers.remove_outlier_speakers import create_clips_without_outliers
from ..protocols import LoggingProtocol, NullLogger
from ..settings.session_settings import SessionSettings

_SETTINGS_FILE = "settings.yaml"


@dataclass
class RemoveOutlierSpeakerClipsCommand:
    speaker_label: str
    output_folder: Path
    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Remove Outlier Speaker Clips"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        settings = SessionSettings.load(common_paths.data_dir() / _SETTINGS_FILE)

        input_dir = common_paths.voice_samples_for_speaker(self.speaker_label)
        if not input_dir.exists():
            raise FileNotFoundError(f"Speaker folder not found: {input_dir}")

        original_count = len(list(input_dir.glob("*.wav")))

        logger.report_message(
            f"[blue]Removing outlier clips for '{self.speaker_label}' "
            f"(min similarity: {settings.min_speaker_similarity})[/blue]"
        )
        create_clips_without_outliers(settings, input_dir, self.output_folder, logger)
        final_count = len(list(self.output_folder.glob("*.wav")))
        logger.report_message(
            f"[green]{original_count} original clip(s) → {final_count} clip(s) written to {self.output_folder}[/green]"
        )
