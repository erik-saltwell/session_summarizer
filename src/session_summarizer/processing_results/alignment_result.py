from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from .process_result_protocol import ProcessResultProtocol


@dataclass
class WordAlignment:
    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float = 0.0  # acoustic confidence [0.0, 1.0]


@dataclass
class AlignmentResult(ProcessResultProtocol):
    words: list[WordAlignment]

    def name(self) -> str:
        return "AlignmentResult"

    def save_to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_from_json(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            words=[WordAlignment(**w) for w in data.get("words", [])],
        )

    def plain_text(self) -> str:
        return " ".join(w.word for w in self.words)

    def get_segments_for_time_range(self, start_time: float, end_time: float) -> list[WordAlignment]:
        return sorted(
            [w for w in self.words if w.end > start_time and w.start < end_time],
            key=lambda w: w.start,
        )
