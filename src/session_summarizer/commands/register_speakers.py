from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..protocols import LoggingProtocol
from .register_speaker import RegisterSpeakerCommand


@dataclass
class RegisterSpeakersCommand:
    """Register all speakers found in the voice_samples directory into registered_speakers.yaml."""

    device: str = "cuda"

    def execute(self, logger: LoggingProtocol) -> None:
        wav_files: list[Path] = sorted(common_paths.voice_samples_dir().glob("*.wav"))
        if not wav_files:
            logger.report_message("[yellow]No WAV files found in voice_samples directory.[/yellow]")
            return

        logger.report_message(f"[blue]Registering {len(wav_files)} speaker(s) from voice_samples/[/blue]")
        for wav_file in wav_files:
            RegisterSpeakerCommand(
                speaker_name=wav_file.stem,
                session_id=None,
                device=self.device,
            ).execute(logger)

    def name(self) -> str:
        return "Register All Speakers"
