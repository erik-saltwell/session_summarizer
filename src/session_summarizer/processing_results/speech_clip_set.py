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
    words: list[WordAlignment] | None = None

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

    def _set_merge_start_properties(self, first: SpeechClip) -> None:
        self.start_time = first.start_time

    def _set_merge_end_properties(self, last: SpeechClip) -> None:
        self.end_time = last.end_time
        self.end_of_turn_probability = last.end_of_turn_probability
        self.set_flag(SpeechClipFlags.END_OF_TURN, last.has_flag(SpeechClipFlags.END_OF_TURN))

    def _set_merge_base_properties(self, other: SpeechClip) -> None:
        speakers: set[str] = set()
        speakers |= self.speakers if not self.speakers == _ANONYMOUS_SPEAKER_SET else set()
        speakers |= other.speakers if not other.speakers == _ANONYMOUS_SPEAKER_SET else set()
        if len(speakers) == 0:
            speakers.add(_ANONYMOUS_SPEAKER)
        self.speakers = speakers

        words: list[WordAlignment] | None
        words = []
        words.extend(self.words if self.words else [])
        words.extend(other.words if other.words else [])
        if len(words) == 0:
            words = None
        self.words = words

        identity = self.identity if self.identity else other.identity
        self.identity = identity

        embedding = None  # embedding needs to be recomputed after merge
        self.embedding = embedding

        self.compute_word_derived_values()

    def merge(self, other: SpeechClip) -> None:
        if other is self:
            return

        first: SpeechClip = self if self.start_time <= other.start_time else other
        last: SpeechClip = self if self.end_time >= other.end_time else other

        self._set_merge_base_properties(other)
        self._set_merge_start_properties(first)
        self._set_merge_end_properties(last)

    def add_word(self, word: WordAlignment) -> None:
        if self.words is None:
            self.words = []
        self.words.append(word)

    def compute_word_derived_values(self) -> None:
        if not self.words:
            self.text = ""
            self.confidence_avg = 0.0
        else:
            sorted_words = sorted(self.words, key=lambda w: (w.start_time, w.end_time))
            self.text = " ".join(w.word for w in sorted_words)
            self.confidence_avg = sum(w.confidence for w in sorted_words) / len(sorted_words)

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
                speakers: str
                if clip.identity is None:
                    speakers = ", ".join(sorted(clip.speakers))
                else:
                    speakers = clip.identity

                flags = " ".join(
                    flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                )
                flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                start_str = f"{clip.start_time: 0.5f}".strip()
                end_str = f"{clip.end_time: 0.5f}".strip()

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
                "words": [
                    {
                        "word": w.word,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "confidence": w.confidence,
                    }
                    for w in clip.words
                ]
                if clip.words is not None
                else None,
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
            raw_words = item.get("words")
            words = (
                [
                    WordAlignment(
                        word=w["word"],
                        start_time=w["start_time"],
                        end_time=w["end_time"],
                        confidence=w.get("confidence", 0.0),
                    )
                    for w in raw_words
                ]
                if raw_words is not None
                else None
            )
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
                words=words,
            )
            instance.append(clip)
        return instance
