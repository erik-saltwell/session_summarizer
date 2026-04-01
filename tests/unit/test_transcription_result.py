from __future__ import annotations

import json
from pathlib import Path

from session_summarizer.processing_results import (
    TranscriptionResult,
    TranscriptionSegment,
)


def _make_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(text="Hello world", start=0.0, end=1.5, confidence=0.95),
            TranscriptionSegment(text="How are you", start=1.8, end=3.2, confidence=0.88),
        ],
        full_text="Hello world How are you",
    )


class TestSave:
    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.json"
        _make_result().save_to_json(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["full_text"] == "Hello world How are you"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world"

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        original = _make_result()
        path = tmp_path / "transcript.json"
        original.save_to_json(path)

        loaded = TranscriptionResult.load_from_json(path)

        assert loaded == original


class TestLoad:
    def test_load_round_trips_with_asdict(self, tmp_path: Path) -> None:
        original = _make_result()

        path = tmp_path / "transcript.json"
        original.save_to_json(path)

        loaded = TranscriptionResult.load_from_json(path)

        assert loaded.full_text == original.full_text
        assert len(loaded.segments) == len(original.segments)
        for loaded_seg, orig_seg in zip(loaded.segments, original.segments, strict=True):
            assert loaded_seg.text == orig_seg.text
            assert loaded_seg.start == orig_seg.start
            assert loaded_seg.end == orig_seg.end
            assert loaded_seg.confidence == orig_seg.confidence

    def test_load_empty_segments(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.json"
        path.write_text(json.dumps({"full_text": "some text", "segments": []}), encoding="utf-8")

        loaded = TranscriptionResult.load_from_json(path)

        assert loaded.full_text == "some text"
        assert loaded.segments == []

    def test_load_missing_optional_fields_uses_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.json"
        path.write_text(json.dumps({}), encoding="utf-8")

        loaded = TranscriptionResult.load_from_json(path)

        assert loaded.full_text == ""
        assert loaded.segments == []
