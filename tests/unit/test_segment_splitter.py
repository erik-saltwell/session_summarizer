from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from session_summarizer.vad.nemo_vad_detector import SpeechSegment, VadResult
from session_summarizer.vad.segment_splitter import (
    SegmentSplitResult,
    compute_segments,
)


def _make_vad(
    speech_ranges: list[tuple[float, float]],
    total_duration: float,
) -> VadResult:
    """Build a VadResult from a list of (start, end) speech ranges."""
    segments: list[SpeechSegment] = []
    prev_end = 0.0
    for start, end in sorted(speech_ranges):
        if start > prev_end:
            segments.append(SpeechSegment(start=prev_end, end=start, is_speech=False))
        segments.append(SpeechSegment(start=start, end=end, is_speech=True))
        prev_end = end
    if prev_end < total_duration:
        segments.append(SpeechSegment(start=prev_end, end=total_duration, is_speech=False))
    return VadResult(segments=segments, audio_duration=total_duration)


class TestShortAudio:
    """Audio shorter than max_length produces a single segment."""

    def test_single_segment_no_cuts(self) -> None:
        vad = _make_vad([(0, 50)], total_duration=50)
        result = compute_segments(vad, min_length=30, max_length=120)
        assert len(result.segments) == 1
        assert len(result.cut_points) == 0
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 50.0

    def test_audio_exactly_max_length(self) -> None:
        vad = _make_vad([(0, 120)], total_duration=120)
        result = compute_segments(vad, min_length=30, max_length=120)
        assert len(result.segments) == 1
        assert len(result.cut_points) == 0


class TestNormalSegmentation:
    """Audio with natural silence gaps is split at silence centers."""

    def test_two_segments_with_silence_gap(self) -> None:
        # 0-60 speech, 60-62 silence, 62-180 speech
        vad = _make_vad([(0, 60), (62, 180)], total_duration=180)
        result = compute_segments(vad, min_length=30, max_length=120)

        assert len(result.cut_points) == 1
        cut = result.cut_points[0]
        assert cut.time == pytest.approx(61.0)  # center of 60-62
        assert cut.silence_duration == pytest.approx(2.0)

        assert len(result.segments) == 2
        assert result.segments[0].start == pytest.approx(0.0)
        assert result.segments[0].end == pytest.approx(61.0)
        assert result.segments[1].start == pytest.approx(61.0)
        assert result.segments[1].end == pytest.approx(180.0)

    def test_prefers_widest_silence_gap(self) -> None:
        # Multiple silence gaps — should pick the widest one in the window
        # 0-40 speech, 40-41 narrow gap, 41-70 speech, 70-75 wide gap, 75-200 speech
        vad = _make_vad([(0, 40), (41, 70), (75, 200)], total_duration=200)
        result = compute_segments(vad, min_length=30, max_length=120)

        # First cut should be at center of the 70-75 gap (wider), not 40-41
        assert len(result.cut_points) >= 1
        first_cut = result.cut_points[0]
        assert first_cut.time == pytest.approx(72.5)  # center of 70-75
        assert first_cut.silence_duration == pytest.approx(5.0)

    def test_multiple_cuts_for_long_audio(self) -> None:
        # 0-80 speech, 80-82 silence, 82-180 speech, 180-183 silence, 183-300 speech
        vad = _make_vad([(0, 80), (82, 180), (183, 300)], total_duration=300)
        result = compute_segments(vad, min_length=30, max_length=120)

        assert len(result.cut_points) == 2
        assert len(result.segments) == 3
        # All segments should be within bounds
        for seg in result.segments:
            assert seg.duration <= 120.0 + 0.01


class TestForceCut:
    """Continuous speech with no silence forces a hard cut."""

    def test_force_cut_no_silence(self) -> None:
        # 250s of continuous speech, no silence
        vad = _make_vad([(0, 250)], total_duration=250)
        result = compute_segments(vad, min_length=30, max_length=120)

        assert len(result.cut_points) >= 1
        # First cut should be at max_length
        assert result.cut_points[0].time == pytest.approx(120.0)
        assert result.cut_points[0].silence_duration == 0.0
        # All segments should respect max_length
        for seg in result.segments:
            assert seg.duration <= 120.0 + 0.01


class TestMinLengthRespected:
    """Cuts that would create segments shorter than min_length are skipped."""

    def test_skip_too_early_silence(self) -> None:
        # 0-10 speech, 10-12 silence, 12-80 speech, 80-82 silence, 82-200 speech
        vad = _make_vad([(0, 10), (12, 80), (82, 200)], total_duration=200)
        result = compute_segments(vad, min_length=30, max_length=120)

        # The 10-12 silence gap should be skipped (only 11s from start)
        # First cut should be at center of 80-82 gap
        assert result.cut_points[0].time == pytest.approx(81.0)


class TestSpeechRatio:
    """speech_ratio is computed correctly for each segment."""

    def test_speech_ratio_calculation(self) -> None:
        # 0-50 speech, 50-60 silence, 60-110 speech, total 150s
        vad = _make_vad([(0, 50), (60, 110)], total_duration=150)
        result = compute_segments(vad, min_length=30, max_length=120)

        # Single segment (150 <= max_length is false, so it may split)
        # Actually 150 > 120, so it should split at center of 50-60 gap = 55
        assert len(result.cut_points) == 1
        seg0 = result.segments[0]
        # Segment 0: 0-55, speech 0-50 = 50/55 ≈ 0.909
        assert seg0.speech_ratio == pytest.approx(50.0 / 55.0, abs=0.01)


class TestEdgeCases:
    """Various edge cases."""

    def test_all_silence(self) -> None:
        vad = VadResult(
            segments=[SpeechSegment(start=0, end=200, is_speech=False)],
            audio_duration=200,
        )
        result = compute_segments(vad, min_length=30, max_length=120)
        assert len(result.segments) >= 1
        for seg in result.segments:
            assert seg.speech_ratio == pytest.approx(0.0)

    def test_very_short_audio(self) -> None:
        vad = _make_vad([(0, 5)], total_duration=5)
        result = compute_segments(vad, min_length=30, max_length=120)
        assert len(result.segments) == 1
        assert len(result.cut_points) == 0


class TestSaveLoad:
    """SegmentSplitResult round-trips through JSON."""

    def test_round_trip(self) -> None:
        vad = _make_vad([(0, 80), (82, 200)], total_duration=200)
        original = compute_segments(vad, min_length=30, max_length=120)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            original.save(path)
            loaded = SegmentSplitResult.load(path)

            assert loaded.audio_duration == original.audio_duration
            assert len(loaded.segments) == len(original.segments)
            assert len(loaded.cut_points) == len(original.cut_points)
            assert len(loaded.speech_segments) == len(original.speech_segments)

            for orig, load in zip(original.segments, loaded.segments, strict=False):
                assert orig.start == pytest.approx(load.start)
                assert orig.end == pytest.approx(load.end)
        finally:
            path.unlink(missing_ok=True)
