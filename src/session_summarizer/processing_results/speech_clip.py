from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag

from .alignment_result import WordAlignment
from .segment_protocol import (
    SegmentProtocol,
    compute_duration_inside_meaningful_boundaries,
    compute_gap_distance,
    compute_overlap,
)

_ANONYMOUS_SPEAKER = "anonymous"
_ANONYMOUS_SPEAKER_SET: set[str] = set({_ANONYMOUS_SPEAKER})


class SpeechClipFlags(IntFlag):
    NONE = 0
    IS_BACKCHANNEL = 1


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
    words: list[WordAlignment] | None = None
    utterance_id: str | None = None

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
    def is_backchannel(self) -> bool:
        return self.has_flag(SpeechClipFlags.IS_BACKCHANNEL)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        if self.words is None:
            return 0
        return len(self.words)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

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
    def create_from_word(cls, word: WordAlignment, speakers: set[str] | None = None) -> SpeechClip:
        result: SpeechClip = cls(
            start_time=word.start_time,
            end_time=word.end_time,
            speakers={_ANONYMOUS_SPEAKER},
            text="",
            words=[word],
        )
        if speakers:
            result.speakers = speakers
        return result

    def merge_with_word(self, word: WordAlignment) -> None:
        self.start_time = min(self.start_time, word.start_time)
        self.end_time = max(self.end_time, word.end_time)
        self.add_word(word)

    def _set_merge_start_properties(self, first: SpeechClip) -> None:
        self.start_time = first.start_time

    def _set_merge_end_properties(self, last: SpeechClip) -> None:
        self.end_time = last.end_time

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
