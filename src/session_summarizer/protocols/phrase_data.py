from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, order=True)
class TextPhrase:
    start_offset: int
    length: int

    @property
    def end_offset(self) -> int:
        return self.start_offset + self.length

    def contains_start(self, start_to_test: int) -> bool:
        return start_to_test >= self.start_offset and start_to_test < self.end_offset

    def contains(self, start_to_test: int, end_to_validate: int) -> bool:
        if self.contains_start(start_to_test):
            if not (end_to_validate >= self.start_offset and end_to_validate < self.end_offset):
                raise RuntimeError("Start is contained but end is not.")
            else:
                return True
        return False


class TextPhraseBuilder(Protocol):
    def build_phrase_data(self, start_offset: int) -> TextPhrase: ...


class TextPhraseSet(Protocol):
    def plain_text(self) -> str: ...
    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]: ...
    def phrase_seperator_length(self) -> int: ...
