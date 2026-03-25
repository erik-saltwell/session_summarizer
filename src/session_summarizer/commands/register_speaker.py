from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

import session_summarizer.utils.common_paths as common_paths

from ..protocols import LoggingProtocol, NullLogger
from ..speaker.eres2netv2_embedder import ERes2NetV2Embedder

REGISTERED_SPEAKERS_FILE = "registered_speakers.yaml"


@dataclass
class RegisterSpeakerCommand:
    """Extract an ERes2NetV2 speaker embedding and store it in registered_speakers.yaml."""

    session_id: str
    speaker_name: str
    wav_file: Path
    embedder: ERes2NetV2Embedder
    logger: LoggingProtocol = NullLogger()

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger

        if not self.wav_file.exists():
            raise FileNotFoundError(f"WAV file not found: {self.wav_file}")

        yaml_path = common_paths.session_path(self.session_id) / REGISTERED_SPEAKERS_FILE

        # Load existing registry or start fresh
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as f:
                data: dict = yaml.safe_load(f) or {}
        else:
            data = {}

        action = "Updating" if self.speaker_name in data else "Registering"
        self.logger.report_message(f"[blue]{action} speaker '{self.speaker_name}'...[/blue]")

        embedding = self.embedder.extract(self.wav_file, logger)

        data[self.speaker_name] = {"embedding": embedding}

        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        self.logger.report_message(
            f"[green]Speaker '{self.speaker_name}' saved to {yaml_path} ({len(embedding)}-dim embedding).[/green]"
        )
