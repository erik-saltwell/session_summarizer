from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestPhrase:
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class TestMeeting:
    phrases: list[TestPhrase] = field(default_factory=list)

    @classmethod
    def load_test_meeting(cls) -> TestMeeting:
        meeting_path = Path("/home/eriksalt/proj/session_summarizer/test_meeting/SimpleTranscript.json")
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
