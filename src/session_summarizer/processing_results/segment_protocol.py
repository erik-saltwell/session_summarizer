from __future__ import annotations

from typing import Protocol


class SegmentProtocol(Protocol):
    @property
    def start_time(self) -> float: ...  # seconds

    @property
    def end_time(self) -> float: ...  # seconds


def compute_overlap(first: SegmentProtocol, second: SegmentProtocol, minimum_overlap: float) -> float:
    # Treat near-zero intersections as zero to stabilize boundary ties.
    inter = min(first.end_time, second.end_time) - max(first.start_time, second.start_time)
    return inter if inter > minimum_overlap else 0.0


def compute_gap_distance(first: SegmentProtocol, second: SegmentProtocol, minimum_overlap: float) -> float:
    if second.end_time <= (first.start_time + minimum_overlap):
        return first.start_time - second.end_time
    if first.end_time <= (second.start_time + minimum_overlap):
        return second.start_time - first.end_time
    return 0.0
