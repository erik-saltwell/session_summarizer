from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from session_summarizer.utils import common_paths

from ..protocols.phrase_data import TextPhrase, TextPhraseBuilder, TextPhraseSet


@dataclass
class TestPhrase(TextPhraseBuilder):
    start: float
    end: float
    text: str
    speaker: str

    def build_phrase_data(self, start_offset: int) -> TextPhrase:
        return TextPhrase(start_offset, len(self.text))


@dataclass
class TestMeeting(TextPhraseSet):
    phrases: list[TestPhrase] = field(default_factory=list)

    @classmethod
    def load_test_meeting(cls) -> TestMeeting:
        meeting_path = common_paths.test_transcript_path()
        payload = json.loads(meeting_path.read_text())

        return cls(
            phrases=[
                TestPhrase(
                    start=phrase["start"],
                    end=phrase["end"],
                    text=phrase["text"],
                    speaker=phrase["speaker"],
                )
                for phrase in payload["phrases"]
            ]
        )

    def phrase_separator_length(self) -> int:
        return 1

    @classmethod
    def phrase_separator(cls) -> str:
        return " "

    def plain_text(self) -> str:
        return TestMeeting.phrase_separator().join(phrase.text for phrase in self.phrases)

    def phrase_builders_in_order(self) -> Iterator[TextPhraseBuilder]:
        yield from self.phrases
