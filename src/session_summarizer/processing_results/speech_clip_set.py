from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .alignment_result import WordAlignment
from .process_result_protocol import ProcessResultProtocol
from .segment_protocol import SegmentProtocol, compute_gap_distance, compute_overlap

_ANONYMOUS_SPEAKER = "anonymous"


@dataclass
class SpeechClip:
    start_time: float
    end_time: float
    speakers: set[str]
    confidence_avg: float
    text: str
    identity: str | None = None
    embedding: list[float] | None = None
    words: list[WordAlignment] | None = None  # Used to collect words while processing. Not saved to disk.

    @property
    def word_count(self) -> int:
        if self.words is None:
            return 0
        return len(self.words)

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1

    @classmethod
    def create_from_word(cls, word: WordAlignment) -> SpeechClip:
        return cls(
            start_time=word.start_time,
            end_time=word.end_time,
            speakers={_ANONYMOUS_SPEAKER},
            confidence_avg=0.0,
            text="",
            words=[word],
        )

    def merge_with_word(self, word: WordAlignment) -> None:
        self.start_time = min(self.start_time, word.start_time)
        self.end_time = max(self.end_time, word.end_time)
        self.add_word(word)

    def merge(self, other: SpeechClip) -> None:
        if (
            self.identity is not None
            or self.embedding is not None
            or other.identity is not None
            or other.embedding is not None
            or self.words is not None
            or other.words is not None
        ):
            raise ValueError("Cannot merge clips that have identity, words, or embedding information")

        self.start_time = min(self.start_time, other.start_time)
        self.end_time = max(self.end_time, other.end_time)
        self.speakers = self.speakers | other.speakers
        self.confidence_avg = (
            (self.confidence_avg * self.word_count + other.confidence_avg * other.word_count)
            / (self.word_count + other.word_count)
            if (self.word_count + other.word_count) > 0
            else 0.0
        )
        if len(self.text) > 0 and len(other.text) > 0:
            if self.start_time < other.start_time:
                self.text = (self.text + " " + other.text).strip()
            else:
                self.text = (other.text + " " + self.text).strip()

    def add_word(self, word: WordAlignment) -> None:
        if self.words is None:
            self.words = []
        self.words.append(word)

    def finalize_words(self) -> None:
        if self.words is None:
            return
        sorted_words = sorted(self.words, key=lambda w: (w.start_time, w.end_time))
        self.text = " ".join(w.word for w in sorted_words)
        self.confidence_avg = sum(w.confidence for w in sorted_words) / len(sorted_words)
        self.words = None

    def overlap(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_overlap(self, other, minimum_overlap)

    def gap_distance(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_gap_distance(self, other, minimum_overlap)


class SpeechClipSet(list["SpeechClip"], ProcessResultProtocol):
    def name(self) -> str:
        return "SpeechClipSet"

    def plain_text(self) -> str:
        return " ".join(clip.text for clip in self)

    def save_to_json(self, path: Path) -> None:
        data = [
            {
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

    def add_clip(self, clip: SpeechClip) -> None:
        self.append(clip)

    def extend_clips(self, clips: list[SpeechClip]) -> None:
        self.extend(clips)

    @classmethod
    def load_from_json(cls, path: Path) -> SpeechClipSet:
        with path.open("r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        instance = cls()
        for item in data:
            clip = SpeechClip(
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
