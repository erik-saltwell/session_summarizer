from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..audio.speaker_tools import merge_speaker_clips_to_min_duration
from ..helpers.remove_outlier_speakers import create_clips_without_outliers
from ..protocols import LoggingProtocol, NullLogger
from ..settings.session_settings import SessionSettings
from ..speaker_embeddings.registered_speakers import RegisteredSpeakers

_SETTINGS_FILE = "settings.yaml"


@dataclass
class RegisterSpeakersCommand:
    """Register all speakers found in per-speaker subdirectories of voice_samples/ into registered_speakers.yaml."""

    logger: LoggingProtocol = NullLogger()

    def name(self) -> str:
        return "Register All Speakers"

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        settings = SessionSettings.load(common_paths.data_dir() / _SETTINGS_FILE)
        voice_dir: Path = common_paths.voice_samples_dir()

        # Discover speaker directories (skip hidden dirs)
        speaker_dirs: list[Path] = sorted(d for d in voice_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        if not speaker_dirs:
            logger.report_message("[yellow]No speaker directories found in voice_samples/.[/yellow]")
            return

        speakers: RegisteredSpeakers = RegisteredSpeakers.load(common_paths.build_speakers_file_path())

        for speaker_dir in speaker_dirs:
            speaker_name = speaker_dir.name
            logger.report_message(f"[blue]Processing speaker '{speaker_name}'...[/blue]")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                merged_dir = tmp / "merged"
                filtered_dir = tmp / "filtered"

                merge_speaker_clips_to_min_duration(
                    speaker_dir,
                    merged_dir,
                    settings.speaker_clips.min_duration_seconds,
                    settings.speaker_clips.silence_gap_seconds,
                    logger,
                )

                centroid = create_clips_without_outliers(settings, merged_dir, filtered_dir, logger)
                if centroid is not None:
                    action = "Updated" if speaker_name in speakers else "Registered"
                    speakers[speaker_name] = centroid
                    logger.report_message(f"[green]{action} speaker '{speaker_name}'[/green]")
                else:
                    logger.report_message(f"[yellow]No clips survived for '{speaker_name}' — skipped.[/yellow]")

        speakers.save(common_paths.build_speakers_file_path())
        logger.report_message(
            f"[green]{len(speakers)} speaker(s) saved to {common_paths.build_speakers_file_path()}.[/green]"
        )
