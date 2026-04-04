from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntFlag, auto
from pathlib import Path

from .alignment_result import WordAlignment
from .process_result_protocol import ProcessResultProtocol
from .segment_protocol import (
    SegmentProtocol,
    compute_duration_inside_meaningful_boundaries,
    compute_gap_distance,
    compute_overlap,
)

_ANONYMOUS_SPEAKER = "anonymous"
_ANONYMOUS_SPEAKER_SET: set[str] = set(_ANONYMOUS_SPEAKER)


class SpeechClipFlags(IntFlag):
    NONE = 0
    END_OF_TURN = auto()


@dataclass
class SpeechClip:
    start_time: float
    end_time: float
    speakers: set[str]
    confidence_avg: float
    text: str
    identity: str | None = None
    embedding: list[float] | None = None
    flags: SpeechClipFlags = field(default=SpeechClipFlags.NONE)
    end_of_turn_probability: float | None = None
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

    @property
    def is_anonymous(self) -> bool:
        return self.speakers == _ANONYMOUS_SPEAKER_SET

    def has_flag(self, flag: SpeechClipFlags) -> bool:
        return bool(self.flags & flag)

    def set_flag(self, flag: SpeechClipFlags, is_set: bool) -> None:
        if is_set:
            self.flags |= flag
        else:
            self.flags &= ~flag

    def duration_inside_meaningful_boundaries(self, epsilon: float) -> float:
        return compute_duration_inside_meaningful_boundaries(self, epsilon)

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
        ):
            raise ValueError("Cannot merge clips that have identity or embedding information")

        if self.text and other.text:
            if self.start_time < other.start_time:
                self.text = (self.text + " " + other.text).strip()
            else:
                self.text = (other.text + " " + self.text).strip()

        self.start_time = min(self.start_time, other.start_time)
        self.end_time = max(self.end_time, other.end_time)

        if self.is_anonymous:
            self.speakers = other.speakers
        elif other.is_anonymous:
            pass
        else:
            self.speakers = self.speakers | other.speakers

        self.confidence_avg = (
            (self.confidence_avg * self.word_count + other.confidence_avg * other.word_count)
            / (self.word_count + other.word_count)
            if (self.word_count + other.word_count) > 0
            else 0.0
        )

        if other.words:
            for word in other.words:
                self.add_word(word)

    def add_word(self, word: WordAlignment) -> None:
        if self.words is None:
            self.words = []
        self.words.append(word)

    def finalize_words(self) -> None:
        if self.words is None:
            return
        sorted_words = sorted(self.words, key=lambda w: (w.start_time, w.end_time))
        self.text = " ".join(w.word for w in sorted_words)
        if len(sorted_words) == 0:
            self.confidence_avg = 0.0
        else:
            self.confidence_avg = sum(w.confidence for w in sorted_words) / len(sorted_words)
        self.words = None

    def overlap(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_overlap(self, other, minimum_overlap)

    def gap_distance(self, other: SegmentProtocol, minimum_overlap: float) -> float:
        return compute_gap_distance(self, other, minimum_overlap)

    def expand_bounds_to_include_words(self, epsilon: float, expansion_limit_seconds: float) -> None:
        if self.words is None:
            return
        min_start = min(min(word.start_time for word in self.words), self.start_time)
        max_end = max(max(word.end_time for word in self.words), self.end_time)

        expand_left = self.start_time - min_start
        expand_right = max_end - self.end_time

        if expand_left > epsilon:
            self.start_time = self.start_time - min(expand_left, expansion_limit_seconds)
        if expand_right > epsilon:
            self.end_time = self.end_time + min(expand_right, expansion_limit_seconds)


class SpeechClipSet(list["SpeechClip"], ProcessResultProtocol):
    def name(self) -> str:
        return "SpeechClipSet"

    def plain_text(self) -> str:
        return " ".join(clip.text for clip in self)

    def save_to_human_format(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                speakers = ", ".join(sorted(clip.speakers))

                flags = " ".join(
                    flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                )
                flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                start_str = f"{clip.start_time: 0.3f}".strip()
                end_str = f"{clip.start_time: 0.3f}".strip()

                f.write(f"{speakers}\n")
                f.write(f"({start_str},{end_str}): {flag_str}\n")
                f.write(f"{clip.text}\n")
                f.write("\n")

    def save_to_json(self, path: Path) -> None:
        data = [
            {
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speakers": sorted(clip.speakers),
                "confidence_avg": clip.confidence_avg,
                "text": clip.text,
                "identity": clip.identity,
                "embedding": clip.embedding,
                "flags": int(clip.flags),
                "end_of_turn_probability": clip.end_of_turn_probability,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_clip(self, clip: SpeechClip) -> None:
        self.append(clip)

    def extend_clips(self, clips: list[SpeechClip]) -> None:
        self.extend(clips)

    def sort_clips(self) -> None:
        self.sort(key=lambda c: (c.start_time, c.end_time))

    @classmethod
    def load_from_json(cls, path: Path) -> SpeechClipSet:
        with path.open("r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
        instance = cls()
        for item in data:
            clip = SpeechClip(
                start_time=item["start_time"],
                end_time=item["end_time"],
                speakers=set(item["speakers"]),
                confidence_avg=item["confidence_avg"],
                text=item["text"],
                identity=item.get("identity"),
                embedding=item.get("embedding"),
                flags=SpeechClipFlags(item.get("flags", 0)),
                end_of_turn_probability=item.get("end_of_turn_probability"),
            )
            instance.append(clip)
        return instance
