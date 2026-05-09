from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiarizationSegment:
    speaker: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]


@dataclass
class MergedDiarizationSegment:
    start_time: float  # seconds
    end_time: float  # seconds
    speakers: list[str]  # sorted for determinism

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_multispeaker(self) -> bool:
        return len(self.speakers) > 1


@dataclass
class MergedDiarizationResult:
    segments: list[MergedDiarizationSegment]

    def save(self, path: Path) -> None:
        data = [{"start": s.start_time, "end": s.end_time, "speakers": s.speakers} for s in self.segments]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> MergedDiarizationResult:
        data = json.loads(path.read_text())
        segments = [
            MergedDiarizationSegment(start_time=s["start"], end_time=s["end"], speakers=s["speakers"]) for s in data
        ]
        return cls(segments=segments)


def merge_overlapping_diarization(raw: DiarizationResult) -> MergedDiarizationResult:
    """Convert overlapping per-speaker segments into non-overlapping windows.

    Uses a sweep-line over all unique start/end timestamps. Each consecutive
    pair of breakpoints becomes one output window whose speakers list contains
    every raw segment active (overlapping) during that window.
    """
    if not raw.segments:
        return MergedDiarizationResult(segments=[])

    breakpoints = sorted({t for seg in raw.segments for t in (seg.start, seg.end)})

    merged: list[MergedDiarizationSegment] = []
    for i in range(len(breakpoints) - 1):
        t_start = breakpoints[i]
        t_end = breakpoints[i + 1]
        active = sorted({seg.speaker for seg in raw.segments if seg.start < t_end and seg.end > t_start})
        if active:
            merged.append(MergedDiarizationSegment(start_time=t_start, end_time=t_end, speakers=active))

    return MergedDiarizationResult(segments=merged)
