from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import session_summarizer.utils.common_paths as common_paths

from ..protocols import EmbeddingFactory, LoggingProtocol, NullLogger
from ..speaker_embeddings import get_embeddings_factory
from ..speaker_embeddings.registered_speakers import RegisteredSpeakers


@dataclass
class RegisterSpeakerCommand:
    """Extract an ERes2NetV2 speaker embedding and store it in registered_speakers.yaml."""

    device: str
    speaker_name: str
    session_id: str | None = None
    logger: LoggingProtocol = NullLogger()

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        wav_file: Path = common_paths.voice_sample_wav_file(self.speaker_name)
        if not wav_file.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        embedder: EmbeddingFactory = get_embeddings_factory(self.device)
        yaml_path: Path = common_paths.build_speakers_file_path(self.session_id)

        speakers: RegisteredSpeakers = RegisteredSpeakers.load(yaml_path)

        action = "Updating" if self.speaker_name in speakers else "Registering"
        self.logger.report_message(f"[blue]{action} speaker '{self.speaker_name}'...[/blue]")

        speakers[self.speaker_name] = embedder.extract(wav_file, logger)
        speakers.save(yaml_path)

        self.logger.report_message(f"[green]Speaker '{self.speaker_name}' saved to {yaml_path} .[/green]")
