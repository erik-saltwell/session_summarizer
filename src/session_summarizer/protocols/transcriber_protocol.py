from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, Self

from .logging_protocol import LoggingProtocol


@dataclass
class TranscriptionSegment:
    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float  # acoustic confidence [0.0, 1.0]


@dataclass
class TranscriptionResult:
    segments: list[TranscriptionSegment] = field(default_factory=list)
    full_text: str = ""

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            segments=[TranscriptionSegment(**seg) for seg in data.get("segments", [])],
            full_text=data.get("full_text", ""),
        )


class TranscriberProtocol(Protocol):
    def transcribe(self, audio_path: Path, logger: LoggingProtocol) -> TranscriptionResult: ...
