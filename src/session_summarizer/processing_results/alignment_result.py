from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from ..protocols import TextPhrase, TextPhraseBuilder
from .process_result_protocol import ProcessResultProtocol
from .segment_protocol import (
    SegmentProtocol,
    compute_gap_distance,
)


@dataclass
class WordAlignment(TextPhraseBuilder):
    word: str
    start_time: float  # seconds
    end_time: float  # seconds
    ground_truth: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def gap_distance(self, other: SegmentProtocol, minimum_overlap: float = 0.0) -> float:
        return compute_gap_distance(self, other, minimum_overlap)

    def build_phrase_data(self, start_offset: int) -> TextPhrase:
        return TextPhrase(start_offset, len(self.word))


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

    def sort(self) -> None:
        self.words.sort(key=lambda w: (w.start_time, w.end_time))
