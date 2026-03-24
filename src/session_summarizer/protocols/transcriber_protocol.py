from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .logging_protocol import LoggingProtocol


@dataclass
class TranscriptionSegment:
    text: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class TranscriptionResult:
    segments: list[TranscriptionSegment] = field(default_factory=list)
    full_text: str = ""


class TranscriberProtocol(Protocol):
    def transcribe(self, audio_path: Path, logger: LoggingProtocol) -> TranscriptionResult: ...
