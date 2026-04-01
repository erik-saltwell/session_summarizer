from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from .process_result_protocol import ProcessResultProtocol
from .segment_protocol import SegmentProtocol, compute_gap_distance, compute_overlap


@dataclass
class WordAlignment:
    word: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 0.0  # acoustic confidence [0.0, 1.0]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2

    def overlap(self, other: SegmentProtocol, minimum_overlap: float = 0.0) -> float:
        return compute_overlap(self, other, minimum_overlap)

    def gap_distance(self, other: SegmentProtocol, minimum_overlap: float = 0.0) -> float:
        return compute_gap_distance(self, other, minimum_overlap)


_DEBUG_TIMING_THRESHOLD = 25.0  # seconds, for debug print statements about word-segment alignment


def word_is_contained_in(word: WordAlignment, start_time: float, end_time: float) -> bool:
    if word.start_time > end_time:
        return False
    debug_str = f"  '{word.word}'({word.start_time:.4f}-{word.end_time:.4f}):seg({start_time:.4f}-{end_time:.4f})"
    return_value: bool = False
    if word.start_time >= start_time and word.end_time <= end_time:
        debug_str += " word inside segment"
        return_value = True
    elif word.start_time < start_time and word.end_time > end_time:
        debug_str += " segment inside word"
        return_value = True
    elif word.start_time >= start_time and word.start_time <= end_time:
        debug_str += " word starts in segment"
        return_value = True

    if start_time < _DEBUG_TIMING_THRESHOLD:
        print(debug_str)

    return return_value

    # midpoint = (word.start + word.end) / 2
    # return start_time <= midpoint < end_time


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

    def get_words_for_time_range(self, start_time: float, end_time: float) -> list[WordAlignment]:
        if start_time < _DEBUG_TIMING_THRESHOLD:
            print(f"({start_time:.4f}-{end_time:.4f})")

        return sorted(
            [w for w in self.words if word_is_contained_in(w, start_time, end_time)],
            key=lambda w: w.start_time,
        )
