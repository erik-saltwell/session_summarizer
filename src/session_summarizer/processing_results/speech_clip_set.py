from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntFlag, auto
from pathlib import Path

from ..protocols import TextPhraseBuilder, TextPhraseSet
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
    text: str
    cosine_similarity: float | None = None
    similarity_residual: float | None = None
    identity: str | None = None
    embedding: list[float] | None = None
    flags: SpeechClipFlags = field(default=SpeechClipFlags.NONE)
    end_of_turn_probability: float | None = None
    words: list[WordAlignment] | None = None

    @property
    def error_formatted_text(self) -> str:
        if not self.words or not self.identity:
            return self.text
        new_words: list[str] = []
        sorted_words = sorted(self.words, key=lambda w: (w.start_time, w.end_time))
        for word in sorted_words:
            if word.ground_truth is None:
                new_words.append("```" + word.word + "```")
            elif word.ground_truth.lower() == self.identity.lower():
                new_words.append("_" + word.word + "_")
            else:
                new_words.append("**" + word.word + "**")

        result: str = " ".join(new_words)
        return result

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
    def confidence_avg(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def wder(self) -> float:
        if self.words is None or self.identity is None:
            return 1.0
        total_words: float = 0.0
        incorrect_words: float = 0.0
        for word in self.words:
            if word.ground_truth is None:
                continue
            total_words += 1.0
            if not self.identity.lower() == word.ground_truth.lower():
                incorrect_words += 1.0
        if total_words == 0.0:
            return 1.0
        return incorrect_words / total_words

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1

    @property
    def is_anonymous(self) -> bool:
        return self.speakers == _ANONYMOUS_SPEAKER_SET

    def compute_speaker_from_pair(self, other: SpeechClip, epsilon: float) -> str:
        if self.identity is not None:
            return self.identity

        intersection = self.speakers & other.speakers
        if len(intersection) > 0:
            return min(intersection)
        return min(self.speakers)

    def compute_speaker(self, prior: SpeechClip | None, next: SpeechClip | None, epsilon: float) -> str:
        if self.identity is not None:
            return self.identity
        if prior is None and next is None:
            return min(self.speakers)
        if prior is None:
            assert next is not None
            return self.compute_speaker_from_pair(next, epsilon)
        if next is None:
            assert prior is not None
            return self.compute_speaker_from_pair(prior, epsilon)

        set_with_prior = self.speakers & prior.speakers
        set_with_next = self.speakers & next.speakers
        set_of_all = set_with_prior & set_with_next
        gap_with_prior = prior.gap_distance(self, epsilon)
        gap_with_next = self.gap_distance(next, epsilon)
        next_is_closer: bool = gap_with_next < gap_with_prior
        set_withh_closer = set_with_next if next_is_closer else set_with_prior
        if len(set_of_all) == 1:
            return min(set_of_all)

        if len(set_with_prior) == 0:
            if len(set_with_next) == 0:
                return min(self.speakers)
            elif len(set_with_next) == 1:
                return min(set_with_next)
            else:
                return min(set_with_next)
        elif len(set_with_prior) == 1:
            if len(set_with_next) == 0:
                return min(set_with_prior)
            elif len(set_with_next) == 1:
                return min(set_withh_closer)
            else:
                return min(set_withh_closer)
        else:
            if len(set_with_next) == 0:
                return min(set_with_prior)
            elif len(set_with_next) == 1:
                return min(set_withh_closer)
            else:
                if len(set_of_all) > 0:
                    return min(set_of_all)
                else:
                    return min(set_with_next | set_with_prior)

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
        else:
            sorted_words = sorted(self.words, key=lambda w: (w.start_time, w.end_time))
            self.text = " ".join(w.word for w in sorted_words)

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

    def sort_words(self) -> None:
        if self.words is not None:
            self.words.sort(key=lambda k: (k.start_time, k.end_time))


class SpeechClipSet(list["SpeechClip"], ProcessResultProtocol, TextPhraseSet):
    def name(self) -> str:
        return "SpeechClipSet"

    def plain_text(self) -> str:
        return " ".join(clip.text for clip in self)

    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]:
        return self.all_words_in_order()

    def all_words_in_order(self) -> Iterator[WordAlignment]:
        self.sort_clips()
        for clip in self:
            if clip.words is None:
                continue
            clip.sort_words()
            yield from clip.words

    def phrase_separator_length(self) -> int:
        return 1

    def save_to_human_format(self, path: Path, include_details: bool = False) -> None:
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                speakers: str
                if clip.identity is None:
                    speakers = ", ".join(sorted(clip.speakers))
                else:
                    speakers = clip.identity

                f.write(f"{speakers}\n")
                if include_details:
                    flags = " ".join(
                        flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                    )
                    flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                    start_str = f"{clip.start_time: 0.5f}".strip()
                    end_str = f"{clip.end_time: 0.5f}".strip()
                    f.write(f"({start_str},{end_str}): {flag_str}\n")
                f.write(f"{clip.text}\n")
                f.write("\n")

    def save_to_error_formatted_text(self, path: Path, include_details: bool = False) -> None:
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                speakers: str
                if clip.identity is None:
                    speakers = ", ".join(sorted(clip.speakers))
                else:
                    speakers = clip.identity

                f.write(f"{speakers}\n")
                if include_details:
                    flags = " ".join(
                        flag.name for flag in SpeechClipFlags if flag and clip.has_flag(flag) and flag.name is not None
                    )
                    flag_str = f"[{flags if flags else 'NO_FLAGS'}]"
                    start_str = f"{clip.start_time: 0.5f}".strip()
                    end_str = f"{clip.end_time: 0.5f}".strip()
                    f.write(f"({start_str},{end_str}): {flag_str}\n")
                f.write(f"{clip.error_formatted_text}\n")
                f.write("\n")

    def _speaker_label(self, clip: SpeechClip) -> str:
        if clip.identity:
            return clip.identity
        return "+".join(sorted(clip.speakers))

    def save_to_rttm(self, path: Path, file_id: str | None = None) -> None:
        fid = file_id if file_id is not None else path.stem
        with path.open("w", encoding="utf-8") as f:
            for clip in self:
                dur = clip.end_time - clip.start_time
                if dur <= 0:
                    continue
                speaker = self._speaker_label(clip)
                f.write(f"SPEAKER {fid} 1 {clip.start_time:.6f} {dur:.6f} <NA> <NA> {speaker} <NA> <NA>\n")

    def save_to_seglst(self, path: Path, session_id: str | None = None) -> None:
        sid = session_id if session_id is not None else path.stem
        data = [
            {
                "session_id": sid,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speaker": self._speaker_label(clip),
                "words": clip.text,
            }
            for clip in self
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_to_json(self, path: Path) -> None:
        data = [
            {
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "speakers": sorted(clip.speakers),
                "text": clip.text,
                "identity": clip.identity,
                "embedding": clip.embedding,
                "flags": int(clip.flags),
                "cosine_similarity": clip.cosine_similarity,
                "similarity_residual": clip.similarity_residual,
                "end_of_turn_probability": clip.end_of_turn_probability,
                "words": [
                    {
                        "word": w.word,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "confidence": w.confidence,
                        "ground_truth": w.ground_truth,
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
    def load_from_test_meeting(cls) -> SpeechClipSet:
        import session_summarizer.utils.common_paths as common_paths

        data = json.loads(common_paths.test_transcript_path().read_text(encoding="utf-8"))
        instance = cls()
        for phrase in data["phrases"]:
            speaker: str = phrase["speaker"]
            instance.append(
                SpeechClip(
                    start_time=float(phrase["start"]),
                    end_time=float(phrase["end"]),
                    speakers={speaker},
                    text=phrase["text"],
                    identity=speaker,
                )
            )
        return instance

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
                        ground_truth=w.get("ground_truth", None),
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
                text=item["text"],
                identity=item.get("identity"),
                embedding=item.get("embedding"),
                cosine_similarity=item.get("cosine_similarity"),
                similarity_residual=item.get("similarity_residual"),
                flags=SpeechClipFlags(item.get("flags", 0)),
                end_of_turn_probability=item.get("end_of_turn_probability"),
                words=words,
            )
            instance.append(clip)
        return instance
