from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .alignment_result import WordAlignment
from .process_result_protocol import ProcessResultProtocol
from .segment_protocol import SegmentProtocol, compute_gap_distance, compute_overlap


class SpeechClipSet(list["SpeechClip"], ProcessResultProtocol):
    def name(self) -> str:
        return "SpeechClipSet"

    def plain_text(self) -> str:
        return " ".join(clip.text for clip in self)

    def save_to_json(self, path: Path) -> None:
        data = [
            {
                "clip_id": clip.clip_id,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speakers": clip.speakers,
                "confidence_avg": clip.confidence_avg,
                "text": clip.text,
                "identity": clip.identity,
                "embedding": clip.embedding,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path: Path) -> SpeechClipSet:
        with path.open("r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        instance = cls()
        for item in data:
            clip = SpeechClip(
                clip_id=item["clip_id"],
                parent=instance,
                start_time=item["start_time"],
                end_time=item["end_time"],
                speakers=item["speakers"],
                confidence_avg=item["confidence_avg"],
                text=item["text"],
                identity=item.get("identity"),
                embedding=item.get("embedding"),
            )
            instance.append(clip)
        return instance


@dataclass
class SpeechClip:
    clip_id: int
    parent: SpeechClipSet
    start_time: float
    end_time: float
    speakers: list[str]
    confidence_avg: float
    text: str
    identity: str | None = None
    embedding: list[float] | None = None
    words: list[WordAlignment] | None = None  # Used to collect words while processing. Not saved to disk.

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_start_clip(self) -> bool:
        return self.clip_id == 0

    @property
    def is_end_clip(self) -> bool:
        return self.clip_id >= len(self.parent) - 1

    @property
    def previous_clip(self) -> SpeechClip | None:
        if self.is_start_clip:
            raise ValueError("Clip ID is 0, cannot determine previous clip")
        return self.parent[self.clip_id - 1]

    @property
    def next_clip(self) -> SpeechClip | None:
        if self.is_end_clip:
            raise ValueError("Clip ID is at the end of the list, cannot determine next clip")
        return self.parent[self.clip_id + 1]

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1

    def add_word(self, word: WordAlignment) -> None:
        if self.words is None:
            self.words = []
        self.words.append(word)

    def finalize_words(self) -> None:
        if self.words is None:
            return
        self.text = " ".join(w.word for w in self.words)
        self.confidence_avg = sum(w.confidence for w in self.words) / len(self.words)
        self.words = None

    def overlap(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_overlap(self, other, minimum_overlap)

    def gap_distance(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_gap_distance(self, other, minimum_overlap)
