from __future__ import annotations

from session_summarizer.diarization.diarizen_diarizer import (
    DiarizationResult,
    DiarizationSegment,
    MergedDiarizationSegment,
    merge_overlapping_diarization,
)


def _raw(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


def _result(*segs: DiarizationSegment) -> DiarizationResult:
    return DiarizationResult(segments=list(segs))


class TestMergeOverlappingDiarization:
    def test_empty_input(self) -> None:
        result = merge_overlapping_diarization(_result())
        assert result.segments == []

    def test_user_example(self) -> None:
        """Exact example from the spec: speakers 1/2/3 with partial overlaps."""
        raw = _result(
            _raw("1", 10.0, 20.0),
            _raw("2", 18.0, 25.0),
            _raw("3", 17.0, 19.0),
        )
        merged = merge_overlapping_diarization(raw)
        expected = [
            MergedDiarizationSegment(start=10.0, end=17.0, speakers=["1"]),
            MergedDiarizationSegment(start=17.0, end=18.0, speakers=["1", "3"]),
            MergedDiarizationSegment(start=18.0, end=19.0, speakers=["1", "2", "3"]),
            MergedDiarizationSegment(start=19.0, end=20.0, speakers=["1", "2"]),
            MergedDiarizationSegment(start=20.0, end=25.0, speakers=["2"]),
        ]
        assert merged.segments == expected

    def test_no_overlaps(self) -> None:
        """Non-overlapping segments pass through unchanged."""
        raw = _result(
            _raw("A", 0.0, 5.0),
            _raw("B", 5.0, 10.0),
            _raw("C", 10.0, 15.0),
        )
        merged = merge_overlapping_diarization(raw)
        assert len(merged.segments) == 3
        assert merged.segments[0] == MergedDiarizationSegment(start=0.0, end=5.0, speakers=["A"])
        assert merged.segments[1] == MergedDiarizationSegment(start=5.0, end=10.0, speakers=["B"])
        assert merged.segments[2] == MergedDiarizationSegment(start=10.0, end=15.0, speakers=["C"])

    def test_full_overlap_two_speakers(self) -> None:
        """Two segments with identical boundaries → one window with both speakers."""
        raw = _result(
            _raw("X", 1.0, 4.0),
            _raw("Y", 1.0, 4.0),
        )
        merged = merge_overlapping_diarization(raw)
        assert merged.segments == [MergedDiarizationSegment(start=1.0, end=4.0, speakers=["X", "Y"])]

    def test_speakers_are_sorted(self) -> None:
        """Speaker list in each window is deterministically sorted."""
        raw = _result(
            _raw("Z", 0.0, 2.0),
            _raw("A", 0.0, 2.0),
            _raw("M", 0.0, 2.0),
        )
        merged = merge_overlapping_diarization(raw)
        assert merged.segments[0].speakers == ["A", "M", "Z"]

    def test_no_gaps_between_adjacent_windows(self) -> None:
        """Output windows are contiguous — no gaps or duplicate timestamps."""
        raw = _result(
            _raw("1", 0.0, 10.0),
            _raw("2", 5.0, 15.0),
        )
        merged = merge_overlapping_diarization(raw)
        for a, b in zip(merged.segments, merged.segments[1:], strict=False):
            assert a.end == b.start

    def test_single_segment(self) -> None:
        raw = _result(_raw("solo", 3.0, 7.0))
        merged = merge_overlapping_diarization(raw)
        assert merged.segments == [MergedDiarizationSegment(start=3.0, end=7.0, speakers=["solo"])]
