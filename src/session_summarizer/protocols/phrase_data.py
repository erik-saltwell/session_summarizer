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


class TextPhraseBuilder(Protocol):
    def build_phrase_data(self, start_offset: int) -> TextPhrase: ...


class TextPhraseSet(Protocol):
    def plain_text(self) -> str: ...
    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]: ...
    def phrase_separator_length(self) -> int: ...
