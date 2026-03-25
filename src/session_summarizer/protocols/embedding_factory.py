from pathlib import Path
from typing import Protocol

from .logging_protocol import LoggingProtocol


class EmbeddingFactory(Protocol):
    def extract(self, audio_path: Path, logger: LoggingProtocol) -> list[float]: ...
